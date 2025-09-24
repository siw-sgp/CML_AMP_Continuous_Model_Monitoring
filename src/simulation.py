# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
# ###########################################################################

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)

# CDSW metrics replacement
import cdsw as metrics

from src.utils import scale_prices
from src.api import ApiUtility
from src.inference import ThreadedModelRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_file = "logs/simulation.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


class Simulation:
    """Simulation class for production-style model monitoring."""

    def __init__(self, model_name: str, dev_mode: bool = False):
        self.api = ApiUtility()
        self.latest_deployment_details = self.api.get_latest_deployment_details(
            model_name=model_name
        )
        self.tmr = ThreadedModelRequest(self.latest_deployment_details)
        self.master_id_uuid_mapping = {}
        self.dev_mode = dev_mode
        self.sample_size = 0.05 if self.dev_mode else 0.8

    def run_simulation(self, train_df, prod_df):
        self.set_simulation_clock(prod_df, months_in_batch=1)

        train_df, prod_df = [
            self.sample_dataframe(df, self.sample_size) for df in (train_df, prod_df)
        ]

        logger.info("------- Starting Section: Train Data -------")
        train_inference_metadata = self.make_inference(train_df)
        formatted_metadata = self.format_metadata_for_delayed_metrics(train_df, is_train=True)
        self.add_delayed_metrics(*formatted_metadata)
        train_metrics_df = self.query_model_metrics(
            **{k: train_inference_metadata[k] for k in train_inference_metadata if k != "id_uuid_mapping"}
        )
        logger.info("------- Finished Section: Train Data -------")

        for i, date_range in tqdm(enumerate(self.date_ranges), total=len(self.date_ranges) + 1):
            formatted_date_range = " <--> ".join([ts.strftime("%m-%d-%Y") for ts in date_range])
            logger.info(f"------- Starting Section {i+1}/{len(self.date_ranges)}: Prod Data ({formatted_date_range})-------")

            new_listings_df = prod_df.loc[prod_df.date_listed.between(date_range[0], date_range[1], inclusive="left")]
            inference_metadata = self.make_inference(new_listings_df)

            formatted_metadata = self.format_metadata_for_delayed_metrics(prod_df, date_range, is_train=False)
            self.add_delayed_metrics(*formatted_metadata)

            metrics_df = self.query_model_metrics()
            new_sold_metrics_df = metrics_df[metrics_df.predictionUuid.isin(formatted_metadata[0])]

            self.build_evidently_reports(
                reference_df=train_metrics_df,
                current_df=new_sold_metrics_df,
                current_date_range=date_range,
            )

            app_name = "Price Regressor Monitoring Dashboard"
            if i == 0:
                self.api.deploy_monitoring_application(application_name=app_name)
            else:
                self.api.restart_running_application(application_name=app_name)

            logger.info(f"------- Finished Section {i+1}/{len(self.date_ranges)}: Prod Data ({formatted_date_range})-------")

    def make_inference(self, df):
        records = self.cast_date_as_str_for_json(df).to_dict(orient="records")
        metadata = self.tmr.threaded_call(records)
        self.master_id_uuid_mapping.update(metadata["id_uuid_mapping"])
        logger.info(f'Made inference and updated the master_id_uuid_mapping with {len(metadata["id_uuid_mapping"])} records')
        return metadata

    def set_simulation_clock(self, prod_df, months_in_batch=1):
        total_months = int(np.ceil((prod_df.date_sold.max() - prod_df.date_sold.min()) / np.timedelta64(1, "M")))
        date_ranges = [
            [prod_df.date_sold.min() + DateOffset(months=n), prod_df.date_sold.min() + DateOffset(months=n + months_in_batch)]
            for n in range(0, total_months, months_in_batch)
        ]
        date_ranges[0][0] = date_ranges[0][0] - DateOffset(years=1)
        logger.info(f"Simulation clock set with {len(date_ranges)} batches")
        self.date_ranges = date_ranges

    def format_metadata_for_delayed_metrics(self, df, date_range=None, is_train=False):
        if not is_train:
            assert date_range is not None
            new_sold_records = df.loc[df.date_sold.between(date_range[0], date_range[1], inclusive="left")]
        else:
            new_sold_records = df.copy()

        uuids = new_sold_records.id.apply(lambda x: self.master_id_uuid_mapping[x]).tolist()
        gts = df[df.id.isin(new_sold_records.id)].price.tolist()
        sold_dates = df[df.id.isin(new_sold_records.id)].date_sold.astype(str).tolist()

        return uuids, gts, sold_dates

    def add_delayed_metrics(self, uuids, ground_truths, sold_dates):
        if len(uuids) != len(ground_truths) != len(sold_dates):
            raise ValueError("UUIDs, ground_truths, and sold_dates must be of same length and correspond by index.")
        for uuid, gt, ds in zip(uuids, ground_truths, sold_dates):
            metrics.track_delayed_metrics(metrics={"ground_truth": gt, "date_sold": ds}, prediction_uuid=uuid)
        logger.info(f"Successfully added ground truth values to {len(uuids)} records")

    def query_model_metrics(self, **kwargs):
        ipt = {"model_deployment_crn": self.latest_deployment_details["latest_deployment_crn"]}
        if kwargs:
            ipt.update(kwargs)
        return self.format_model_metrics_query(metrics.read_metrics(**ipt))

    @staticmethod
    def sample_dataframe(df, fraction):
        return df.sample(frac=fraction, random_state=42)

    @staticmethod
    def cast_date_as_str_for_json(df):
        for column, dt in zip(df.columns, df.dtypes):
            if dt.type not in [np.int64, np.float64]:
                df.loc[:, column] = df.loc[:, column].astype(str)
        return df

    @staticmethod
    def format_model_metrics_query(metrics_dict: Dict):
        metrics_df = pd.json_normalize(metrics_dict["metrics"])
        return metrics_df[[col for col in metrics_df.columns if col.split(".")[0] == "metrics"] + ["predictionUuid"]].rename(
            columns={col: col.split(".")[-1] for col in metrics_df.columns}
        )

    @staticmethod
    def build_evidently_reports(reference_df, current_df, current_date_range):
        TARGET = "ground_truth"
        PREDICTION = "predicted_result"
        NUM_FEATURES = ["sqft_living", "sqft_lot", "sqft_above"]
        CAT_FEATURES = ["waterfront", "zipcode", "condition", "view", "bedrooms", "bathrooms"]

        column_mapping = ColumnMapping()
        column_mapping.target = TARGET
        column_mapping.prediction = PREDICTION
        column_mapping.numerical_features = NUM_FEATURES
        column_mapping.categorical_features = CAT_FEATURES

        report_dir = os.path.join(
            "apps/static/reports/",
            f'{current_date_range[0].strftime("%m-%d-%Y")}_{current_date_range[1].strftime("%m-%d-%Y")}',
        )
        os.makedirs(report_dir, exist_ok=True)

        reports = [
            ("data_drift", DataDriftTab()),
            ("num_target_drift", NumTargetDriftTab()),
            ("reg_performance", RegressionPerformanceTab()),
        ]

        for report_name, tab in reports:
            dashboard = Dashboard(tabs=[tab])
            dashboard.calculate(
                reference_data=scale_prices(reference_df)
                .sample(n=len(current_df), random_state=42)
                .set_index("date_sold", drop=True)
                .sort_index()
                .round(2),
                current_data=scale_prices(current_df)
                .set_index("date_sold", drop=True)
                .sort_index()
                .round(2),
                column_mapping=column_mapping,
            )
            report_path = os.path.join(report_dir, f"{report_name}_report.html")
            dashboard.save(report_path)
            logger.info(f"Generated new Evidently report: {report_path}")
