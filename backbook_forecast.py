#!/usr/bin/env python3
"""
Complete Backbook Forecasting Model

This module forecasts loan portfolio performance (collections, GBV, impairment, NBV)
for 12-36 months using historical rate curves and impairment assumptions.

Usage:
    python backbook_forecast.py --fact-raw Fact_Raw_Full.csv --methodology Rate_Methodology.csv

Author: Claude Code
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the backbook forecast model."""

    MAX_MONTHS: int = 12  # Default forecast horizon
    LOOKBACK_PERIODS: int = 6  # Default lookback for CohortAvg
    MOB_THRESHOLD: int = 3  # Minimum MOB for rate calculation

    # Debt Sale Configuration
    # Coverage ratio - percentage of provisions covering debt sale pool
    DS_COVERAGE_RATIO: float = 0.785  # 78.5%
    # Proceeds rate - pence received per £1 of GBV sold
    DS_PROCEEDS_RATE: float = 0.24  # 24p per £1
    # Debt sale months - calendar months when debt sales occur (quarterly)
    DS_MONTHS: List[int] = [3, 6, 9, 12]  # March, June, September, December

    # Debt Sale Forecast Redesign (DSDonorCRScaled approach)
    # (year, month) tuples excluded from calibration and donor construction
    DS_OUTLIER_MONTHS: List[Tuple[int, int]] = [(2025, 12)]
    # Number of most recent non-outlier DS event quarters to use when calibrating Layer B
    DS_CALIBRATION_N_QUARTERS: int = 3
    # Minimum post-sale DS observations a donor cohort must have to be eligible
    DS_MIN_OBS_DONOR: int = 2
    # Shrinkage target: alpha = min(1, N_own_obs / DS_SHRINKAGE_TARGET)
    DS_SHRINKAGE_TARGET: int = 6
    # DS rate cap multiplier: cap = DS_RATE_CAP_MULTIPLIER × segment historical max event rate
    DS_RATE_CAP_MULTIPLIER: float = 1.5
    # MOB window used to compute CR profile vectors for donor similarity
    CR_SIM_MOB_START: int = 6
    CR_SIM_MOB_END: int = 24

    # Rate caps by metric - wide ranges to accommodate data variance
    # Caps are sanity checks, not business rules. CohortAvg/methodology drive values.
    RATE_CAPS: Dict[str, Tuple[float, float]] = {
        'Coll_Principal': (-0.50, 0.15),  # Usually negative (collections), some positive variance
        'Coll_Interest': (-0.20, 0.05),   # Usually negative (collections), some positive variance
        'InterestRevenue': (0.0, 0.50),   # Always positive
        'WO_DebtSold': (0.0, 0.20),       # Always positive
        'WO_Other': (0.0, 0.05),          # Always positive
        'ContraSettlements_Principal': (-0.15, 0.01),  # Usually negative
        'ContraSettlements_Interest': (-0.01, 0.01),   # Usually negative
        'NewLoanAmount': (0.0, 1.0),      # Always positive
        'Total_Coverage_Ratio': (0.0, 2.50),  # Allow up to 250% coverage
        'Debt_Sale_Coverage_Ratio': (0.50, 1.00),
        'Debt_Sale_Proceeds_Rate': (0.10, 1.00),
    }

    # Valid segments
    SEGMENTS: List[str] = ['NON PRIME', 'NRP-S', 'NRP-M', 'NRP-L', 'PRIME']

    # Metrics for rate calculation
    METRICS: List[str] = [
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ContraSettlements_Principal',
        'ContraSettlements_Interest', 'NewLoanAmount',
        'Total_Coverage_Ratio', 'Debt_Sale_Coverage_Ratio',
        'Debt_Sale_Proceeds_Rate'
    ]

    # Valid rate calculation approaches
    VALID_APPROACHES: List[str] = [
        'CohortAvg', 'CohortTrend', 'DonorCohort', 'ShapeBorrowScaled',
        'SegMedian', 'Manual', 'Zero', 'ScaledCohortAvg',
        'StaticCohortAvg', 'DSDonorCRScaled',
    ]

    # Seasonality configuration
    ENABLE_SEASONALITY: bool = False  # Disable seasonal adjustment for coverage ratios
    SEASONALITY_METRIC: str = 'Total_Coverage_Ratio'  # Metric to apply seasonality to

    # Overlay configuration
    ENABLE_OVERLAYS: bool = True  # Enable overlay adjustments
    OVERLAY_FILE: str = 'sample_data/Overlays.csv'  # Path to overlay configuration file


# =============================================================================
# SECTION 2: HELPER FUNCTIONS
# =============================================================================

def parse_date(date_val: Any) -> pd.Timestamp:
    """
    Parse date value to pandas Timestamp.

    Handles both M/D/YYYY and MM/DD/YYYY formats.

    Args:
        date_val: Date value to parse (string, datetime, or Timestamp)

    Returns:
        pd.Timestamp: Parsed date
    """
    if pd.isna(date_val):
        return pd.NaT
    if isinstance(date_val, pd.Timestamp):
        return date_val
    if isinstance(date_val, datetime):
        return pd.Timestamp(date_val)

    # Try parsing as string
    try:
        return pd.to_datetime(date_val, format='%m/%d/%Y')
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(date_val, format='%Y-%m-%d')
        except (ValueError, TypeError):
            try:
                return pd.to_datetime(date_val)
            except Exception:
                logger.warning(f"Could not parse date: {date_val}")
                return pd.NaT


def end_of_month(date: pd.Timestamp) -> pd.Timestamp:
    """
    Get the last day of the month for a given date.

    Args:
        date: Input date

    Returns:
        pd.Timestamp: Last day of the month
    """
    if pd.isna(date):
        return pd.NaT
    return date + pd.offsets.MonthEnd(0)


def clean_cohort(cohort_val: Any) -> str:
    """
    Clean cohort value to string format.

    Args:
        cohort_val: Cohort value (int, float, or string)

    Returns:
        str: Cleaned cohort string (YYYYMM format)
    """
    if pd.isna(cohort_val):
        return ''
    if isinstance(cohort_val, (int, float)):
        return str(int(cohort_val))
    cohort_str = str(cohort_val).replace('.0', '').strip()
    return cohort_str


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero

    Returns:
        float: Result of division or default
    """
    if pd.isna(denominator) or denominator == 0:
        return default
    if pd.isna(numerator):
        return default
    result = numerator / denominator
    if np.isinf(result) or np.isnan(result):
        return default
    return result


def is_debt_sale_month(date: pd.Timestamp) -> bool:
    """
    Check if a calendar month is a debt sale month.

    Debt sales occur quarterly: March, June, September, December.

    Args:
        date: Calendar date (Timestamp)

    Returns:
        bool: True if this is a debt sale month
    """
    if pd.isna(date):
        return False
    return date.month in Config.DS_MONTHS


# =============================================================================
# SECTION 2B: RAW DATA TRANSFORMER
# =============================================================================
# Transforms raw Fact_Raw_New.xlsx format into model-ready format
# Replicates Power Query M code transformations

import re
from calendar import monthrange
from datetime import date as dt_date

# Column renaming map (raw → model format)
RAW_COLUMN_RENAME_MAP = {
    'cohort': 'Cohort_Raw',
    'calendarmonth': 'CalendarMonth_Raw',
    'lob': 'LOB',
    'loansize': 'LoanSize',
    'openinggbv': 'OpeningGBV',
    'disbursalsexcltopup': 'Disb_ExclTopups',
    'disbursalstopup': 'TopUp_IncrCash',
    'loanamount': 'NewLoanAmount',
    'principalcollections': 'Coll_Principal',
    'interestcollections': 'Coll_Interest',
    'principalcontrasettlement': 'ContraSettlements_Principal',
    'nonprincipalcontrasettlement': 'ContraSettlements_Interest',
    'debtsalewriteoffs': 'WO_DebtSold',
    'otherwriteoffs': 'WO_Other',
    'closinggbv': 'ClosingGBV_Reported',
    'interestrevenue': 'InterestRevenue',
    'provisionatmonthend': 'Provision_Balance',
    'debtsaleproceeds': 'Debt_Sale_Proceeds',
}

# Numeric columns for aggregation
RAW_NUMERIC_COLUMNS = [
    'OpeningGBV', 'Disb_ExclTopups', 'TopUp_IncrCash', 'NewLoanAmount',
    'Coll_Principal', 'Coll_Interest', 'ContraSettlements_Principal',
    'ContraSettlements_Interest', 'WO_DebtSold', 'WO_Other',
    'ClosingGBV_Reported', 'InterestRevenue', 'Provision_Balance', 'Debt_Sale_Proceeds',
]


def yyyymm_to_eom(yyyymm: int) -> dt_date:
    """Convert YYYYMM integer to end-of-month date."""
    year = yyyymm // 100
    month = yyyymm % 100
    _, last_day = monthrange(year, month)
    return dt_date(year, month, last_day)


def parse_cohort_ym(cohort_val) -> int:
    """Parse cohort value to YYYYMM integer. Returns -1 for PRE-2020."""
    if pd.isna(cohort_val):
        return None
    cohort_str = str(cohort_val).strip().upper()
    if 'PRE' in cohort_str and '2020' in cohort_str:
        return -1
    try:
        return int(float(cohort_val))
    except (ValueError, TypeError):
        pass
    try:
        dt = pd.to_datetime(cohort_val)
        return dt.year * 100 + dt.month
    except Exception:
        return None


def get_cohort_cluster(cohort_ym: int) -> int:
    """
    Map cohort YYYYMM to clustered cohort based on Backbook groupings.
    - PRE-2020 (-1) → 201912
    - 202001-202012 → 202001 (Backbook 4)
    - 202101-202208 → 202101 (Backbook 3)
    - 202209-202305 → 202201 (Backbook 2)
    - 202306-202403 → 202301 (Backbook 1)
    - Others → keep original (monthly cohorts from 202404+)
    """
    if cohort_ym is None:
        return None
    if cohort_ym == -1:
        return 201912
    if 202001 <= cohort_ym <= 202012:
        return 202001
    if 202101 <= cohort_ym <= 202208:
        return 202101
    if 202209 <= cohort_ym <= 202305:
        return 202201
    if 202306 <= cohort_ym <= 202403:
        return 202301
    return cohort_ym


def parse_loan_size_bucket(loan_size: str) -> str:
    """Parse loan size string to S/M/L bucket."""
    if pd.isna(loan_size):
        return ''
    raw = re.sub(r'[^0-9\-]', '', str(loan_size))
    parts = raw.split('-')
    if len(parts) < 2:
        return ''
    try:
        low = int(parts[0])
        high = int(parts[1])
    except (ValueError, IndexError):
        return ''
    if low < 5:
        return 'S'
    elif low >= 5 and high <= 15:
        return 'M'
    elif low >= 15:
        return 'L'
    return ''


def build_segment_from_lob(lob: str, loan_size: str) -> str:
    """Build segment from LOB and LoanSize."""
    if pd.isna(lob):
        return ''
    lob_clean = str(lob).strip().upper().replace('-', ' ')
    if lob_clean == 'NEAR PRIME':
        size_bucket = parse_loan_size_bucket(loan_size)
        if size_bucket:
            return f'NRP-{size_bucket}'
        return 'NEAR PRIME'
    return lob_clean


def calculate_mob_from_dates(calendar_month_raw: int, cohort_date: dt_date) -> int:
    """Calculate Months on Book from cohort date to calendar month."""
    cal_year = calendar_month_raw // 100
    cal_month = calendar_month_raw % 100
    return (cal_year * 12 + cal_month) - (cohort_date.year * 12 + cohort_date.month)


def transform_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Fact_Raw_New data into model-ready format.
    Replicates Power Query M code transformations.
    """
    logger.info(f"Starting transformation of {len(df_raw)} raw rows...")

    # Step 1: Rename columns
    df = df_raw.rename(columns=RAW_COLUMN_RENAME_MAP).copy()

    # Step 2: Parse cohort YYYYMM
    df['CohortYM'] = df['Cohort_Raw'].apply(parse_cohort_ym)

    # Step 3: Create CohortDate (original, not clustered)
    def cohort_ym_to_date(ym):
        if ym is None:
            return None
        if ym == -1:
            return dt_date(2019, 12, 31)
        return dt_date(ym // 100, ym % 100, 1)

    df['CohortDate'] = df['CohortYM'].apply(cohort_ym_to_date)

    # Step 4: Apply cohort clustering
    df['CohortCluster'] = df['CohortYM'].apply(get_cohort_cluster)
    df['Cohort'] = df['CohortCluster'].astype(str)
    logger.info("Applied cohort clustering (Backbook 1-4)")

    # Step 5: Build Segment from LOB + LoanSize
    df['Segment'] = df.apply(
        lambda row: build_segment_from_lob(row.get('LOB'), row.get('LoanSize')), axis=1
    )

    # Step 6: Calculate MOB from original CohortDate
    df['MOB'] = df.apply(
        lambda row: calculate_mob_from_dates(row['CalendarMonth_Raw'], row['CohortDate'])
        if row['CohortDate'] is not None else None, axis=1
    )

    # Filter out negative MOB
    df = df[df['MOB'] >= 0].copy()
    logger.info(f"Calculated MOB, {len(df)} rows with MOB >= 0")

    # Step 7: Convert CalendarMonth to end-of-month date
    df['CalendarMonth'] = df['CalendarMonth_Raw'].apply(
        lambda x: pd.Timestamp(yyyymm_to_eom(x))
    )

    # Step 8: Add DaysInMonth
    df['DaysInMonth'] = df['CalendarMonth'].dt.days_in_month

    # Step 9: Fill missing numeric values with 0
    for col in RAW_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Step 10: Group by CalendarMonth, Cohort, Segment, MOB
    group_cols = ['CalendarMonth', 'Cohort', 'Segment', 'MOB']
    agg_dict = {col: 'sum' for col in RAW_NUMERIC_COLUMNS if col in df.columns}
    agg_dict['DaysInMonth'] = 'mean'

    df_grouped = df.groupby(group_cols, as_index=False).agg(agg_dict)
    df_grouped['DaysInMonth'] = df_grouped['DaysInMonth'].round().astype(int)

    logger.info(f"Grouped to {len(df_grouped)} rows by Cohort × Segment × CalendarMonth × MOB")

    # Sort and return
    df_grouped = df_grouped.sort_values(
        ['Segment', 'Cohort', 'CalendarMonth', 'MOB']
    ).reset_index(drop=True)

    logger.info(f"Unique Segments: {df_grouped['Segment'].unique().tolist()}")
    logger.info(f"Unique Cohorts: {sorted(df_grouped['Cohort'].unique().tolist())}")

    return df_grouped


# =============================================================================
# SECTION 2C: SEASONALITY FUNCTIONS
# =============================================================================
# Functions to calculate, apply, and remove seasonal adjustments from coverage ratios
# This allows us to analyze underlying trends without monthly noise

# Global storage for seasonal factors (calculated once, used throughout)
_seasonal_factors: Dict[str, Dict[int, float]] = {}


def calculate_seasonal_factors(fact_raw: pd.DataFrame, metric: str = 'Total_Coverage_Ratio') -> Dict[str, Dict[int, float]]:
    """
    Calculate seasonal adjustment factors from historical data.

    For each segment, calculates the average coverage ratio by calendar month,
    then computes factors relative to the segment's overall average.

    Factor > 1.0 means that month typically has higher CR than average
    Factor < 1.0 means that month typically has lower CR than average

    Args:
        fact_raw: Historical loan data with CalendarMonth and coverage ratio
        metric: The metric to calculate seasonality for (default: Total_Coverage_Ratio)

    Returns:
        Dict[str, Dict[int, float]]: Nested dict of {Segment: {month: factor}}
    """
    global _seasonal_factors
    logger.info("Calculating seasonal factors for coverage ratios...")

    # First, calculate coverage ratios from the raw data if needed
    # Group by CalendarMonth, Segment to get weighted coverage ratio
    monthly_cr = fact_raw.groupby(['CalendarMonth', 'Segment']).agg({
        'Provision_Balance': 'sum',
        'ClosingGBV_Reported': 'sum'
    }).reset_index()

    monthly_cr['Coverage_Ratio'] = monthly_cr.apply(
        lambda r: safe_divide(r['Provision_Balance'], r['ClosingGBV_Reported']), axis=1
    )

    # Extract calendar month number
    monthly_cr['Month'] = monthly_cr['CalendarMonth'].dt.month

    # Calculate factors by segment
    seasonal_factors = {}

    for segment in monthly_cr['Segment'].unique():
        seg_data = monthly_cr[monthly_cr['Segment'] == segment].copy()

        # Calculate overall segment average CR
        seg_avg = seg_data['Coverage_Ratio'].mean()

        if seg_avg == 0 or pd.isna(seg_avg):
            # If segment has no meaningful data, use neutral factors
            seasonal_factors[segment] = {m: 1.0 for m in range(1, 13)}
            continue

        # Calculate average CR by month for this segment
        month_avg = seg_data.groupby('Month')['Coverage_Ratio'].mean()

        # Calculate factors: month_avg / segment_avg
        factors = {}
        for month in range(1, 13):
            if month in month_avg.index and not pd.isna(month_avg[month]):
                factors[month] = month_avg[month] / seg_avg
            else:
                factors[month] = 1.0  # Neutral if no data

        seasonal_factors[segment] = factors

    # Also calculate an "ALL" segment factor for fallback
    overall_avg = monthly_cr['Coverage_Ratio'].mean()
    if overall_avg > 0 and not pd.isna(overall_avg):
        month_avg_all = monthly_cr.groupby('Month')['Coverage_Ratio'].mean()
        all_factors = {}
        for month in range(1, 13):
            if month in month_avg_all.index and not pd.isna(month_avg_all[month]):
                all_factors[month] = month_avg_all[month] / overall_avg
            else:
                all_factors[month] = 1.0
        seasonal_factors['ALL'] = all_factors
    else:
        seasonal_factors['ALL'] = {m: 1.0 for m in range(1, 13)}

    # Log the calculated factors
    logger.info("Seasonal factors calculated:")
    for seg in ['NON PRIME', 'NRP-S', 'NRP-M', 'NRP-L', 'PRIME', 'ALL']:
        if seg in seasonal_factors:
            factors_str = ", ".join([f"{m}:{v:.3f}" for m, v in sorted(seasonal_factors[seg].items())])
            logger.info(f"  {seg}: {factors_str}")

    # Store globally for later use
    _seasonal_factors = seasonal_factors

    return seasonal_factors


def get_seasonal_factor(segment: str, month: int) -> float:
    """
    Get the seasonal factor for a segment and calendar month.

    Args:
        segment: Segment name
        month: Calendar month (1-12)

    Returns:
        float: Seasonal factor (1.0 = neutral)
    """
    global _seasonal_factors

    if not _seasonal_factors:
        return 1.0

    if segment in _seasonal_factors:
        return _seasonal_factors[segment].get(month, 1.0)
    elif 'ALL' in _seasonal_factors:
        return _seasonal_factors['ALL'].get(month, 1.0)
    else:
        return 1.0


def deseasonalize_coverage_ratio(cr: float, segment: str, month: int) -> float:
    """
    Remove seasonal effect from a coverage ratio.

    De-seasonalized CR = Actual CR / Seasonal Factor

    Args:
        cr: Actual coverage ratio
        segment: Segment name
        month: Calendar month (1-12)

    Returns:
        float: De-seasonalized coverage ratio
    """
    factor = get_seasonal_factor(segment, month)
    if factor == 0 or pd.isna(factor):
        return cr
    return cr / factor


def reseasonalize_coverage_ratio(cr: float, segment: str, month: int) -> float:
    """
    Re-apply seasonal effect to a coverage ratio forecast.

    Seasonalized CR = Base CR × Seasonal Factor

    Args:
        cr: Base (de-seasonalized) coverage ratio forecast
        segment: Segment name
        month: Calendar month (1-12)

    Returns:
        float: Seasonally adjusted coverage ratio
    """
    factor = get_seasonal_factor(segment, month)
    return cr * factor


def add_deseasonalized_cr_to_curves(curves: pd.DataFrame, fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add de-seasonalized coverage ratio column to curves DataFrame.

    This creates a 'Total_Coverage_Ratio_Deseasonalized' column that can be
    used for trend analysis and forecasting without seasonal noise.

    Args:
        curves: Curves DataFrame with Segment, Cohort, MOB, Total_Coverage_Ratio
        fact_raw: Raw data used to get CalendarMonth for each observation

    Returns:
        pd.DataFrame: Curves with added de-seasonalized column
    """
    logger.info("Adding de-seasonalized coverage ratios to curves...")

    # Get the calendar month for each Segment × Cohort × MOB combination
    # from the fact_raw data
    cal_month_lookup = fact_raw.groupby(['Segment', 'Cohort', 'MOB'])['CalendarMonth'].first().reset_index()

    # Merge to get calendar month
    curves_with_month = curves.merge(
        cal_month_lookup,
        on=['Segment', 'Cohort', 'MOB'],
        how='left'
    )

    # Calculate de-seasonalized CR
    def calc_deseas_cr(row):
        if pd.isna(row.get('CalendarMonth')) or pd.isna(row.get('Total_Coverage_Ratio')):
            return row.get('Total_Coverage_Ratio', 0)
        month = row['CalendarMonth'].month if hasattr(row['CalendarMonth'], 'month') else 1
        return deseasonalize_coverage_ratio(row['Total_Coverage_Ratio'], row['Segment'], month)

    curves_with_month['Total_Coverage_Ratio_Deseasonalized'] = curves_with_month.apply(calc_deseas_cr, axis=1)

    # Drop the CalendarMonth column if it wasn't there originally
    if 'CalendarMonth' not in curves.columns:
        curves_with_month = curves_with_month.drop(columns=['CalendarMonth'])

    logger.info("De-seasonalized coverage ratios added successfully")
    return curves_with_month


# =============================================================================
# SECTION 2D: OVERLAY FUNCTIONS
# =============================================================================
# Overlay functionality allows users to apply manual adjustments to forecasted
# OUTPUT METRICS (amounts like collections, impairment, revenue, etc.)
# Overlays are applied AFTER all calculations are complete
# This enables scenario analysis and manual corrections to final outputs

# Global storage for overlay rules
_overlay_rules: List[Dict[str, Any]] = []

# Valid metrics that can be overlayed
OVERLAY_METRICS = [
    'Coll_Principal',
    'Coll_Interest',
    'InterestRevenue',
    'WO_DebtSold',
    'WO_Other',
    'ClosingGBV',
    'Total_Provision_Balance',
    'Gross_Impairment_ExcludingDS',
    'Debt_Sale_Impact',
    'Net_Impairment',
    'ClosingNBV',
]


def load_overlays(filepath: str = None) -> List[Dict[str, Any]]:
    """
    Load overlay rules from CSV file.

    Overlay CSV format:
        Segment,ForecastMonth_Start,ForecastMonth_End,Metric,Type,Value,Explanation

    Type options:
        - Multiply: Amount × Value (e.g., 0.95 = -5%)
        - Add: Amount + Value (e.g., -1000000 = subtract £1m)
        - Replace: Use Value directly

    Metrics that can be overlayed:
        - Coll_Principal, Coll_Interest (collections)
        - InterestRevenue
        - WO_DebtSold, WO_Other (writeoffs)
        - ClosingGBV
        - Total_Provision_Balance
        - Gross_Impairment_ExcludingDS, Debt_Sale_Impact, Net_Impairment
        - ClosingNBV

    Args:
        filepath: Path to overlay CSV file. If None, uses Config.OVERLAY_FILE

    Returns:
        List[Dict]: List of overlay rule dictionaries
    """
    global _overlay_rules

    if filepath is None:
        filepath = Config.OVERLAY_FILE

    if not os.path.exists(filepath):
        logger.info(f"No overlay file found at {filepath}, overlays disabled")
        _overlay_rules = []
        return []

    logger.info(f"Loading overlays from: {filepath}")

    try:
        df = pd.read_csv(filepath, comment='#')

        # Skip empty files
        if len(df) == 0:
            logger.info("Overlay file is empty, no overlays applied")
            _overlay_rules = []
            return []

        rules = []
        for _, row in df.iterrows():
            rule = {
                'Segment': str(row.get('Segment', 'ALL')).strip().upper(),
                'Metric': str(row.get('Metric', '')).strip(),
                'ForecastMonth_Start': pd.to_datetime(row.get('ForecastMonth_Start')) if pd.notna(row.get('ForecastMonth_Start')) else None,
                'ForecastMonth_End': pd.to_datetime(row.get('ForecastMonth_End')) if pd.notna(row.get('ForecastMonth_End')) else None,
                'Type': str(row.get('Type', 'Multiply')).strip().capitalize(),
                'Value': float(row.get('Value', 1.0)),
                'Explanation': str(row.get('Explanation', '')),
            }

            # Validate rule
            if rule['Metric'] and rule['Type'] in ['Multiply', 'Add', 'Replace']:
                if rule['Metric'] not in OVERLAY_METRICS:
                    logger.warning(f"  Unknown overlay metric '{rule['Metric']}', skipping. Valid: {OVERLAY_METRICS}")
                    continue
                rules.append(rule)
                date_range = ''
                if rule['ForecastMonth_Start'] or rule['ForecastMonth_End']:
                    start = rule['ForecastMonth_Start'].strftime('%Y-%m') if rule['ForecastMonth_Start'] else 'start'
                    end = rule['ForecastMonth_End'].strftime('%Y-%m') if rule['ForecastMonth_End'] else 'end'
                    date_range = f" ({start} to {end})"
                logger.info(f"  Loaded overlay: {rule['Metric']} for {rule['Segment']}{date_range} "
                           f"-> {rule['Type']}({rule['Value']})")

        _overlay_rules = rules
        logger.info(f"Loaded {len(rules)} overlay rules")
        return rules

    except Exception as e:
        logger.warning(f"Error loading overlays: {e}")
        _overlay_rules = []
        return []


def apply_metric_overlays(output_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply overlay adjustments to forecast output DataFrame.

    Overlays are applied to final output metrics (amounts), not rates.
    This allows users to adjust collections, impairment, revenue, etc.

    Args:
        output_df: DataFrame with forecast output rows

    Returns:
        pd.DataFrame: DataFrame with overlays applied and tracking columns added
    """
    global _overlay_rules

    if not _overlay_rules or not Config.ENABLE_OVERLAYS:
        return output_df

    df = output_df.copy()

    # Initialize overlay tracking column
    df['Overlays_Applied'] = ''

    for rule in _overlay_rules:
        metric = rule['Metric']
        segment = rule['Segment']
        overlay_type = rule['Type']
        value = rule['Value']

        # Skip if metric not in DataFrame
        if metric not in df.columns:
            continue

        # Build mask for rows to apply overlay
        mask = pd.Series([True] * len(df), index=df.index)

        # Segment filter
        if segment != 'ALL':
            mask = mask & (df['Segment'].str.upper() == segment)

        # Forecast month filter
        if rule['ForecastMonth_Start'] is not None:
            mask = mask & (df['ForecastMonth'] >= rule['ForecastMonth_Start'])
        if rule['ForecastMonth_End'] is not None:
            mask = mask & (df['ForecastMonth'] <= rule['ForecastMonth_End'])

        if not mask.any():
            continue

        # Store original value for tracking
        original_col = f'{metric}_PreOverlay'
        if original_col not in df.columns:
            df[original_col] = df[metric]

        # Apply overlay
        if overlay_type == 'Multiply':
            df.loc[mask, metric] = df.loc[mask, metric] * value
            desc = f"{metric}×{value:.4f}"
        elif overlay_type == 'Add':
            df.loc[mask, metric] = df.loc[mask, metric] + value
            desc = f"{metric}{value:+.2f}"
        elif overlay_type == 'Replace':
            df.loc[mask, metric] = value
            desc = f"{metric}={value:.2f}"
        else:
            continue

        # Track what overlay was applied
        df.loc[mask, 'Overlays_Applied'] = df.loc[mask, 'Overlays_Applied'].apply(
            lambda x: f"{x}; {desc}" if x else desc
        )

        logger.debug(f"Applied overlay: {desc} to {mask.sum()} rows")

    # Log summary
    overlay_rows = (df['Overlays_Applied'] != '').sum()
    if overlay_rows > 0:
        logger.info(f"Applied overlays to {overlay_rows} output rows")

    return df


def get_overlay_rules() -> List[Dict[str, Any]]:
    """Get the currently loaded overlay rules."""
    global _overlay_rules
    return _overlay_rules.copy()


# =============================================================================
# SECTION 3: DATA LOADING FUNCTIONS
# =============================================================================

def load_fact_raw(filepath: str) -> pd.DataFrame:
    """
    Load and validate historical loan data.

    Supports both CSV (.csv) and Excel (.xlsx) file formats.
    Automatically detects and transforms raw format (Fact_Raw_New) vs processed format.
    Automatically maps column names from common variations.

    Args:
        filepath: Path to Fact_Raw file (CSV or Excel)

    Returns:
        pd.DataFrame: Validated fact raw data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading fact raw data from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load based on file extension
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(filepath)
        logger.info(f"Loaded {len(df)} rows from Excel file")
    else:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from CSV file")

    # Detect if this is the raw format (Fact_Raw_New) that needs transformation
    # Raw format has lowercase columns: 'cohort', 'calendarmonth', 'lob', 'loansize'
    raw_format_indicators = ['cohort', 'calendarmonth', 'lob', 'loansize']
    is_raw_format = all(col in df.columns for col in raw_format_indicators)

    if is_raw_format:
        logger.info("Detected raw data format (Fact_Raw_New) - applying transformations...")
        df = transform_raw_data(df)
        logger.info("Successfully transformed raw data to model format")

    # Column name mappings (source -> target)
    # Maps variations found in different data sources to standard names
    column_mappings = {
        'Provision': 'Provision_Balance',
        'DebtSaleProceeds': 'Debt_Sale_Proceeds',
    }

    # Apply column mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            logger.info(f"Renamed column '{old_name}' to '{new_name}'")

    # Required columns (core fields that must exist)
    required_cols = [
        'CalendarMonth', 'Cohort', 'Segment', 'MOB', 'OpeningGBV',
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported', 'DaysInMonth'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse dates (handle both string and datetime formats)
    if not pd.api.types.is_datetime64_any_dtype(df['CalendarMonth']):
        df['CalendarMonth'] = df['CalendarMonth'].apply(parse_date)
    df['CalendarMonth'] = df['CalendarMonth'].apply(end_of_month)

    # Clean cohort (convert to string format YYYYMM)
    df['Cohort'] = df['Cohort'].apply(clean_cohort)

    # Ensure numeric columns
    numeric_cols = [
        'MOB', 'OpeningGBV', 'Coll_Principal', 'Coll_Interest',
        'InterestRevenue', 'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported', 'DaysInMonth'
    ]

    # Optional columns that may or may not exist
    optional_numeric_cols = [
        'NewLoanAmount', 'ContraSettlements_Principal', 'ContraSettlements_Interest'
    ]
    for col in optional_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.info(f"Added missing column {col} with default value 0")

    # Impairment columns (optional, default to 0)
    impairment_cols = [
        'Provision_Balance', 'Debt_Sale_WriteOffs',
        'Debt_Sale_Provision_Release', 'Debt_Sale_Proceeds'
    ]
    for col in impairment_cols:
        if col not in df.columns:
            df[col] = 0.0
            logger.info(f"Added missing column {col} with default value 0")

    # Convert all numeric columns
    all_numeric = numeric_cols + optional_numeric_cols + impairment_cols
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Handle provision balance sign convention
    # Some systems store provisions as negative (liability), convert to positive for calculations
    if 'Provision_Balance' in df.columns:
        if df['Provision_Balance'].sum() < 0:
            logger.info("Converting negative provision balances to positive values")
            df['Provision_Balance'] = df['Provision_Balance'].abs()

    # Ensure MOB is integer
    df['MOB'] = df['MOB'].astype(int)

    # Sort data
    df = df.sort_values(['CalendarMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    # Log summary statistics
    logger.info(f"Validated {len(df)} rows with {df['Cohort'].nunique()} cohorts")
    logger.info(f"Segments: {df['Segment'].unique().tolist()}")
    logger.info(f"Date range: {df['CalendarMonth'].min()} to {df['CalendarMonth'].max()}")
    logger.info(f"MOB range: {df['MOB'].min()} to {df['MOB'].max()}")

    return df


def load_rate_methodology(filepath: str) -> pd.DataFrame:
    """
    Load rate calculation control table.

    Args:
        filepath: Path to Rate_Methodology.csv

    Returns:
        pd.DataFrame: Methodology rules
    """
    logger.info(f"Loading rate methodology from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} methodology rules")

    # Fill NaN with "ALL"
    for col in ['Segment', 'Cohort', 'Metric']:
        if col in df.columns:
            df[col] = df[col].fillna('ALL').astype(str).str.strip()

    # Clean cohort
    df['Cohort'] = df['Cohort'].apply(lambda x: clean_cohort(x) if x != 'ALL' else 'ALL')

    # Ensure MOB range columns are integers
    df['MOB_Start'] = pd.to_numeric(df['MOB_Start'], errors='coerce').fillna(0).astype(int)
    df['MOB_End'] = pd.to_numeric(df['MOB_End'], errors='coerce').fillna(999).astype(int)

    # Clean Approach
    df['Approach'] = df['Approach'].astype(str).str.strip()

    # Clean Param1 and Param2
    if 'Param1' in df.columns:
        df['Param1'] = df['Param1'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    else:
        df['Param1'] = None

    if 'Param2' in df.columns:
        df['Param2'] = df['Param2'].apply(lambda x: str(x).strip() if pd.notna(x) else None)
    else:
        df['Param2'] = None

    # Load Donor_Cohort column (used by DSDonorCRScaled approach)
    if 'Donor_Cohort' in df.columns:
        df['Donor_Cohort'] = df['Donor_Cohort'].apply(
            lambda x: str(x).strip() if pd.notna(x) and str(x).strip() not in ('', 'nan') else None
        )
    else:
        df['Donor_Cohort'] = None

    # Validate approaches
    invalid_approaches = df[~df['Approach'].isin(Config.VALID_APPROACHES)]['Approach'].unique()
    if len(invalid_approaches) > 0:
        logger.warning(f"Found invalid approaches: {invalid_approaches}")

    return df


def load_debt_sale_schedule(filepath: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load debt sale assumptions (optional).

    Args:
        filepath: Path to Debt_Sale_Schedule.csv or None

    Returns:
        pd.DataFrame or None: Debt sale schedule
    """
    if filepath is None or not os.path.exists(filepath):
        logger.info("No debt sale schedule loaded")
        return None

    logger.info(f"Loading debt sale schedule from: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} debt sale entries")

    # Parse dates
    df['ForecastMonth'] = df['ForecastMonth'].apply(parse_date)
    df['ForecastMonth'] = df['ForecastMonth'].apply(end_of_month)

    # Clean cohort
    df['Cohort'] = df['Cohort'].apply(clean_cohort)

    # Ensure numeric columns
    for col in ['Debt_Sale_WriteOffs', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df = df.sort_values(['ForecastMonth', 'Segment', 'Cohort']).reset_index(drop=True)

    return df


# =============================================================================
# SECTION 4: CURVES CALCULATION FUNCTIONS
# =============================================================================

def calculate_curves_base(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate historical rates from actuals.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Base curves with rates by Segment × Cohort × MOB
    """
    logger.info("Calculating base curves...")

    # Group by Segment, Cohort, MOB
    agg_dict = {
        'OpeningGBV': 'sum',
        'NewLoanAmount': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'InterestRevenue': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'ContraSettlements_Principal': 'sum',
        'ContraSettlements_Interest': 'sum',
        'DaysInMonth': 'mean',
        'ClosingGBV_Reported': 'sum',
        'Provision_Balance': 'sum',
        'Debt_Sale_WriteOffs': 'sum',
        'Debt_Sale_Provision_Release': 'sum',
        'Debt_Sale_Proceeds': 'sum',
    }

    curves = fact_raw.groupby(['Segment', 'Cohort', 'MOB']).agg(agg_dict).reset_index()

    # Calculate rates
    curves['NewLoanAmount_Rate'] = curves.apply(
        lambda r: safe_divide(r['NewLoanAmount'], r['OpeningGBV']), axis=1
    )
    curves['Coll_Principal_Rate'] = curves.apply(
        lambda r: safe_divide(r['Coll_Principal'], r['OpeningGBV']), axis=1
    )
    curves['Coll_Interest_Rate'] = curves.apply(
        lambda r: safe_divide(r['Coll_Interest'], r['OpeningGBV']), axis=1
    )
    # Annualize interest revenue rate
    curves['InterestRevenue_Rate'] = curves.apply(
        lambda r: safe_divide(r['InterestRevenue'], r['OpeningGBV']) * safe_divide(365, r['DaysInMonth'], 12),
        axis=1
    )
    curves['WO_DebtSold_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_DebtSold'], r['OpeningGBV']), axis=1
    )
    curves['WO_Other_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_Other'], r['OpeningGBV']), axis=1
    )
    curves['ContraSettlements_Principal_Rate'] = curves.apply(
        lambda r: safe_divide(r['ContraSettlements_Principal'], r['OpeningGBV']), axis=1
    )
    curves['ContraSettlements_Interest_Rate'] = curves.apply(
        lambda r: safe_divide(r['ContraSettlements_Interest'], r['OpeningGBV']), axis=1
    )

    # Calculate coverage ratios
    curves['Total_Coverage_Ratio'] = curves.apply(
        lambda r: safe_divide(r['Provision_Balance'], r['ClosingGBV_Reported']), axis=1
    )
    curves['Debt_Sale_Coverage_Ratio'] = curves.apply(
        lambda r: safe_divide(r['Debt_Sale_Provision_Release'], r['Debt_Sale_WriteOffs']), axis=1
    )
    curves['Debt_Sale_Proceeds_Rate'] = curves.apply(
        lambda r: safe_divide(r['Debt_Sale_Proceeds'], r['Debt_Sale_WriteOffs']), axis=1
    )

    curves = curves.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Calculated curves for {len(curves)} Segment × Cohort × MOB combinations")
    return curves


def extend_curves(curves_base: pd.DataFrame, max_months: int) -> pd.DataFrame:
    """
    Extend curves beyond max observed MOB for forecasting.

    Args:
        curves_base: Base curves with historical rates
        max_months: Number of months to extend

    Returns:
        pd.DataFrame: Extended curves
    """
    logger.info(f"Extending curves for {max_months} months...")

    # Rate columns to extend
    rate_cols = [col for col in curves_base.columns if col.endswith('_Rate')]

    extensions = []

    # Group by Segment and Cohort
    for (segment, cohort), group in curves_base.groupby(['Segment', 'Cohort']):
        max_mob = group['MOB'].max()
        last_row = group[group['MOB'] == max_mob].iloc[0]

        for offset in range(1, max_months + 1):
            new_mob = max_mob + offset
            new_row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': new_mob,
            }
            # Copy rate columns from last MOB
            for col in rate_cols:
                new_row[col] = last_row[col]

            # Copy other columns with defaults
            for col in ['OpeningGBV', 'NewLoanAmount', 'Coll_Principal', 'Coll_Interest',
                        'InterestRevenue', 'WO_DebtSold', 'WO_Other', 'ClosingGBV_Reported',
                        'ContraSettlements_Principal', 'ContraSettlements_Interest',
                        'Provision_Balance', 'Debt_Sale_WriteOffs', 'Debt_Sale_Provision_Release',
                        'Debt_Sale_Proceeds']:
                if col in curves_base.columns:
                    new_row[col] = 0.0

            new_row['DaysInMonth'] = 30

            extensions.append(new_row)

    if extensions:
        extensions_df = pd.DataFrame(extensions)
        curves_extended = pd.concat([curves_base, extensions_df], ignore_index=True)
    else:
        curves_extended = curves_base.copy()

    curves_extended = curves_extended.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Extended curves to {len(curves_extended)} rows")
    return curves_extended


# =============================================================================
# SECTION 5: IMPAIRMENT CURVES FUNCTIONS
# =============================================================================

def calculate_impairment_actuals(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate impairment metrics from historical data.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Impairment actuals
    """
    logger.info("Calculating impairment actuals...")

    # Group by Segment, Cohort, CalendarMonth
    agg_dict = {
        'Provision_Balance': 'sum',
        'ClosingGBV_Reported': 'sum',
        'Debt_Sale_WriteOffs': 'sum',
        'Debt_Sale_Provision_Release': 'sum',
        'Debt_Sale_Proceeds': 'sum',
        'WO_Other': 'sum',
        'MOB': 'max',
    }

    impairment = fact_raw.groupby(['Segment', 'Cohort', 'CalendarMonth']).agg(agg_dict).reset_index()

    # Rename for clarity
    impairment.rename(columns={
        'Provision_Balance': 'Total_Provision_Balance',
        'ClosingGBV_Reported': 'Total_ClosingGBV',
    }, inplace=True)

    # Calculate coverage ratio
    impairment['Total_Coverage_Ratio'] = impairment.apply(
        lambda r: safe_divide(r['Total_Provision_Balance'], r['Total_ClosingGBV']), axis=1
    )

    # Calculate debt sale coverage and proceeds rate
    # Note: These are calculated BEFORE sign convention is applied
    # Using abs() to ensure positive ratios regardless of sign convention
    impairment['Debt_Sale_Coverage_Ratio'] = impairment.apply(
        lambda r: safe_divide(abs(r['Debt_Sale_Provision_Release']), abs(r['Debt_Sale_WriteOffs'])), axis=1
    )
    impairment['Debt_Sale_Proceeds_Rate'] = impairment.apply(
        lambda r: safe_divide(abs(r['Debt_Sale_Proceeds']), abs(r['Debt_Sale_WriteOffs'])), axis=1
    )

    # Sort and calculate provision movement
    impairment = impairment.sort_values(['Segment', 'Cohort', 'CalendarMonth']).reset_index(drop=True)

    impairment['Prior_Provision_Balance'] = impairment.groupby(['Segment', 'Cohort'])['Total_Provision_Balance'].shift(1).fillna(0)
    impairment['Total_Provision_Movement'] = impairment['Total_Provision_Balance'] - impairment['Prior_Provision_Balance']

    # Apply sign convention:
    # - Write-offs (Debt_Sale_WriteOffs, WO_Other): POSITIVE (absolute amounts)
    # - DS_Provision_Release: POSITIVE (income/benefit)
    # - DS_Proceeds: POSITIVE (income/benefit)
    impairment['Debt_Sale_WriteOffs'] = impairment['Debt_Sale_WriteOffs'].abs()  # POSITIVE
    impairment['WO_Other'] = impairment['WO_Other'].abs()  # POSITIVE
    impairment['Debt_Sale_Provision_Release'] = impairment['Debt_Sale_Provision_Release'].abs()  # POSITIVE
    impairment['Debt_Sale_Proceeds'] = impairment['Debt_Sale_Proceeds'].abs()  # POSITIVE

    # Calculate impairment components
    # Non_DS = Total + DS_Release (add back the release to isolate non-DS movement)
    impairment['Non_DS_Provision_Movement'] = impairment['Total_Provision_Movement'] + impairment['Debt_Sale_Provision_Release']
    # Gross impairment = NEGATED provision movement - WO_Other
    # P&L convention: provision increase = charge (negative), provision decrease = release (positive)
    # WO_Other is stored as positive, subtract to represent expense
    impairment['Gross_Impairment_ExcludingDS'] = -impairment['Non_DS_Provision_Movement'] - impairment['WO_Other']
    # Debt_Sale_Impact: -WriteOffs (expense) + Release (positive) + Proceeds (positive)
    impairment['Debt_Sale_Impact'] = (
        -impairment['Debt_Sale_WriteOffs'] +
        impairment['Debt_Sale_Provision_Release'] +
        impairment['Debt_Sale_Proceeds']
    )
    impairment['Net_Impairment'] = impairment['Gross_Impairment_ExcludingDS'] + impairment['Debt_Sale_Impact']

    logger.info(f"Calculated impairment actuals for {len(impairment)} entries")
    return impairment


def calculate_impairment_curves(impairment_actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate impairment rates for forecasting.

    Args:
        impairment_actuals: Impairment actuals data

    Returns:
        pd.DataFrame: Impairment curves with rates
    """
    logger.info("Calculating impairment curves...")

    # Group by Segment, Cohort, MOB
    agg_dict = {
        'Total_Provision_Balance': 'mean',
        'Total_ClosingGBV': 'mean',
        'Total_Coverage_Ratio': 'mean',
        'Debt_Sale_Coverage_Ratio': 'mean',
        'Debt_Sale_Proceeds_Rate': 'mean',
        'WO_Other': 'sum',
    }

    curves = impairment_actuals.groupby(['Segment', 'Cohort', 'MOB']).agg(agg_dict).reset_index()

    # Calculate WO_Other rate
    curves['WO_Other_Rate'] = curves.apply(
        lambda r: safe_divide(r['WO_Other'], r['Total_ClosingGBV']), axis=1
    )

    curves = curves.sort_values(['Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Calculated impairment curves for {len(curves)} entries")
    return curves


# =============================================================================
# SECTION 6: SEED GENERATION FUNCTIONS
# =============================================================================

def generate_seed_curves(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create forecast starting point from last month of actuals.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Seed with 1 row per Segment × Cohort
    """
    logger.info("Generating seed curves...")

    # Get max calendar month
    max_cal = fact_raw['CalendarMonth'].max()
    logger.info(f"Using last month: {max_cal}")

    # Filter to last month
    last_month = fact_raw[fact_raw['CalendarMonth'] == max_cal].copy()

    # Group by Segment, Cohort
    agg_dict = {
        'ClosingGBV_Reported': 'sum',
        'MOB': 'max',
        'Provision_Balance': 'sum',
    }

    seed = last_month.groupby(['Segment', 'Cohort']).agg(agg_dict).reset_index()

    # Rename columns
    seed.rename(columns={
        'ClosingGBV_Reported': 'BoM',
        'Provision_Balance': 'Prior_Provision_Balance',
    }, inplace=True)

    # MOB for forecast is max MOB + 1
    seed['MOB'] = seed['MOB'] + 1

    # Calculate forecast month (max_cal + 1 month)
    seed['ForecastMonth'] = end_of_month(max_cal + relativedelta(months=1))

    # Filter where BoM > 0
    seed = seed[seed['BoM'] > 0].reset_index(drop=True)

    logger.info(f"Generated seed with {len(seed)} Segment × Cohort combinations")
    return seed


def generate_impairment_seed(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Create impairment starting point.

    Args:
        fact_raw: Historical loan data

    Returns:
        pd.DataFrame: Impairment seed
    """
    logger.info("Generating impairment seed...")

    # Get max calendar month
    max_cal = fact_raw['CalendarMonth'].max()

    # Filter to last month
    last_month = fact_raw[fact_raw['CalendarMonth'] == max_cal].copy()

    # Group by Segment, Cohort
    agg_dict = {
        'Provision_Balance': 'sum',
        'ClosingGBV_Reported': 'sum',
    }

    seed = last_month.groupby(['Segment', 'Cohort']).agg(agg_dict).reset_index()

    # Rename columns
    seed.rename(columns={
        'Provision_Balance': 'Prior_Provision_Balance',
        'ClosingGBV_Reported': 'ClosingGBV',
    }, inplace=True)

    # Calculate forecast month
    seed['ForecastMonth'] = end_of_month(max_cal + relativedelta(months=1))

    logger.info(f"Generated impairment seed with {len(seed)} entries")
    return seed


# =============================================================================
# SECTION 7: METHODOLOGY LOOKUP FUNCTIONS
# =============================================================================

def get_specificity_score(row: pd.Series, segment: str, cohort: str, metric: str, mob: int) -> float:
    """
    Calculate specificity score for a methodology rule.

    Scoring:
    - Exact Segment match: +8 points
    - Exact Cohort match: +4 points
    - Exact Metric match: +2 points
    - Narrower MOB range: +1/(1 + MOB_End - MOB_Start) points (tiebreaker)

    Args:
        row: Methodology rule row
        segment: Target segment
        cohort: Target cohort
        metric: Target metric
        mob: Target MOB

    Returns:
        float: Specificity score
    """
    score = 0.0

    # Segment match
    if row['Segment'] == segment:
        score += 8

    # Cohort match
    if row['Cohort'] == cohort:
        score += 4

    # Metric match
    if row['Metric'] == metric:
        score += 2

    # MOB range width (narrower is better)
    mob_range = row['MOB_End'] - row['MOB_Start']
    score += 1 / (1 + mob_range)

    return score


def get_methodology(methodology_df: pd.DataFrame, segment: str, cohort: str,
                   mob: int, metric: str) -> Dict[str, Any]:
    """
    Find best matching rate calculation rule.

    Args:
        methodology_df: Methodology rules DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric: Target metric

    Returns:
        dict: Best matching rule with Approach, Param1, Param2
    """
    cohort_str = clean_cohort(cohort)

    # Filter matching rules
    mask = (
        ((methodology_df['Segment'] == segment) | (methodology_df['Segment'] == 'ALL')) &
        ((methodology_df['Cohort'] == cohort_str) | (methodology_df['Cohort'] == 'ALL')) &
        ((methodology_df['Metric'] == metric) | (methodology_df['Metric'] == 'ALL')) &
        (methodology_df['MOB_Start'] <= mob) &
        (methodology_df['MOB_End'] >= mob)
    )

    matches = methodology_df[mask].copy()

    if len(matches) == 0:
        return {
            'Approach': 'NoMatch_ERROR',
            'Param1': None,
            'Param2': None
        }

    # Calculate specificity scores
    matches['_score'] = matches.apply(
        lambda r: get_specificity_score(r, segment, cohort_str, metric, mob),
        axis=1
    )

    # Get best match
    best_match = matches.loc[matches['_score'].idxmax()]

    return {
        'Approach': best_match['Approach'],
        'Param1': best_match['Param1'],
        'Param2': best_match['Param2'],
        'Donor_Cohort': best_match.get('Donor_Cohort', None),
    }


# =============================================================================
# SECTION 8: RATE CALCULATION FUNCTIONS
# =============================================================================

def fn_cohort_avg(curves_df: pd.DataFrame, segment: str, cohort: str,
                  mob: int, metric_col: str, lookback: int = 6,
                  exclude_zeros: bool = False) -> Optional[float]:
    """
    Calculate average rate from last N MOBs (post-MOB 3).

    IMPORTANT: Only uses historical data (MOB < forecast MOB), not extended curves.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB (the MOB being forecast)
        metric_col: Column name for metric rate
        lookback: Number of periods to look back
        exclude_zeros: If True, only average non-zero rates (for debt sale metrics)

    Returns:
        float or None: Average rate
    """
    cohort_str = clean_cohort(cohort)

    # Filter data - use MOB < mob to only include HISTORICAL data, not extended curves
    # For early MOBs (MOB <= MOB_THRESHOLD + 1), use all available data (MOB >= 1)
    # This fixes the bug where newly originated cohorts got zero rates because
    # there was no data satisfying MOB > 3 AND MOB < 4 (impossible for MOB=4 cohorts)
    if mob <= Config.MOB_THRESHOLD + 1:
        # For early MOBs, use all available historical data
        min_mob_filter = 1  # Include from MOB 1 onwards
    else:
        # For mature MOBs, skip the initial ramp-up period
        min_mob_filter = Config.MOB_THRESHOLD

    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] >= min_mob_filter) &
        (curves_df['MOB'] < mob)  # Only include HISTORICAL data, not forecast MOB
    )

    data = curves_df[mask].sort_values('MOB', ascending=False)

    # For early MOBs, allow single data point; for mature MOBs, require 2+
    min_data_points = 1 if mob <= Config.MOB_THRESHOLD + 1 else 2
    if len(data) < min_data_points:
        return None

    if metric_col not in data.columns:
        return None

    # For debt sale metrics, only average non-zero rates
    # (zeros just mean no debt sale occurred that month)
    if exclude_zeros:
        non_zero_data = data[data[metric_col] > 0]
        if len(non_zero_data) == 0:
            return None
        # Take last N non-zero values
        non_zero_data = non_zero_data.head(lookback)
        rate = non_zero_data[metric_col].mean()
    else:
        # Take last N rows
        data = data.head(lookback)
        rate = data[metric_col].mean()

    if pd.isna(rate):
        return None

    return float(rate)


def fn_static_cohort_avg(curves_df: pd.DataFrame, segment: str, cohort: str,
                         mob: int, metric_col: str, lookback: int = 6,
                         exclude_zeros: bool = False) -> Optional[float]:
    """
    Calculate average rate from historical ACTUAL curve points only.

    Unlike fn_cohort_avg (which can include previously forecasted values in
    rolling mode), this function excludes rows marked as forecast-generated
    (`__is_forecast` == True). This helps prevent forecast compounding drift.

    Args:
        curves_df: Curves DataFrame (may include forecast-augmented rows)
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB (the MOB being forecast)
        metric_col: Column name for metric rate
        lookback: Number of periods to look back
        exclude_zeros: If True, only average non-zero rates

    Returns:
        float or None: Average rate
    """
    cohort_str = clean_cohort(cohort)

    if mob <= Config.MOB_THRESHOLD + 1:
        min_mob_filter = 1
    else:
        min_mob_filter = Config.MOB_THRESHOLD

    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] >= min_mob_filter) &
        (curves_df['MOB'] < mob)
    )

    # Exclude forecast-generated rows when marker is available
    if '__is_forecast' in curves_df.columns:
        mask = mask & (curves_df['__is_forecast'] != True)

    data = curves_df[mask].sort_values('MOB', ascending=False)

    min_data_points = 1 if mob <= Config.MOB_THRESHOLD + 1 else 2
    if len(data) < min_data_points:
        return None

    if metric_col not in data.columns:
        return None

    if exclude_zeros:
        non_zero_data = data[data[metric_col] > 0]
        if len(non_zero_data) == 0:
            return None
        non_zero_data = non_zero_data.head(lookback)
        rate = non_zero_data[metric_col].mean()
    else:
        data = data.head(lookback)
        rate = data[metric_col].mean()

    if pd.isna(rate):
        return None

    return float(rate)


def fn_cohort_trend(curves_df: pd.DataFrame, segment: str, cohort: str,
                    mob: int, metric_col: str) -> Optional[float]:
    """
    Linear regression extrapolation on post-MOB 3 data.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Predicted rate
    """
    cohort_str = clean_cohort(cohort)

    # Filter data - same early MOB handling as fn_cohort_avg
    if mob <= Config.MOB_THRESHOLD + 1:
        min_mob_filter = 1
    else:
        min_mob_filter = Config.MOB_THRESHOLD

    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] >= min_mob_filter) &
        (curves_df['MOB'] < mob)
    )

    data = curves_df[mask].copy()

    if len(data) < 2:
        return None

    if metric_col not in data.columns:
        return None

    x = data['MOB'].values
    y = data[metric_col].values

    # Remove NaN values
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return None

    x = x[valid_mask]
    y = y[valid_mask]

    # Linear regression: y = a + b*x
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return None

    b = (n * sum_xy - sum_x * sum_y) / denominator
    a = (sum_y - b * sum_x) / n

    # Predict at target MOB
    predicted = a + b * mob

    if np.isnan(predicted) or np.isinf(predicted):
        return None

    return float(predicted)


def fn_donor_cohort(curves_df: pd.DataFrame, segment: str, donor_cohort: str,
                    mob: int, metric_col: str) -> Optional[float]:
    """
    Copy rate from donor cohort at same MOB.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        donor_cohort: Donor cohort YYYYMM
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Donor rate
    """
    donor_cohort_str = clean_cohort(donor_cohort)

    # Filter data
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == donor_cohort_str) &
        (curves_df['MOB'] == mob)
    )

    data = curves_df[mask]

    if len(data) == 0:
        return None

    if metric_col not in data.columns:
        return None

    rate = data[metric_col].iloc[0]

    if pd.isna(rate):
        return None

    return float(rate)


def fn_shape_borrow_scaled(curves_df: pd.DataFrame, segment: str, cohort: str,
                           donor_cohort: str, mob: int, metric_col: str,
                           mode: str = 'additive',
                           ref_window: int = 12) -> Dict[str, Any]:
    """
    Borrow curve SHAPE from a donor cohort, anchored to the target cohort's own level.

    Replaces the old ScaledDonor (ratio-only) approach with a flexible implementation
    supporting both additive (de-meaned) and ratio scaling modes.

    ADDITIVE MODE (preferred for principal collections):
        r̂(m) = Lc + (rd(m) - r̄d_ref)
        where:
            Lc     = target cohort's mean rate over the last `ref_window` ACTUAL MOBs
            rd(m)  = donor rate at forecast MOB m
            r̄d_ref = donor's mean rate over the same ref_window MOBs as Lc

        This preserves absolute step sizes (e.g., MOB 36→37 spike in bps)
        and anchors the level to the target's own recent actuals.

    RATIO MODE (for metrics where relative shape is more meaningful):
        r̂(m) = rd(m) × (Lc / rd_ref_single)
        where:
            Lc           = target cohort rate at reference anchor MOB (single point)
            rd_ref_single = donor rate at the same anchor MOB

        Use when the metric has a consistent proportional relationship
        between cohort vintages (e.g., interest revenue yield curves).

    KEY SAFETY: Lc is ALWAYS computed from actuals-only rows (__is_forecast != True).
    This prevents compounding drift from the rolling working_curves window.

    Param2 format in Rate_Methodology CSV:
        "additive:12"  -> additive mode, ref_window = 12 MOBs
        "ratio:6"      -> ratio mode, ref_window/anchor MOB = 6
        "additive"     -> additive mode, default ref_window = 12
        "ratio"        -> ratio mode, default ref_window = 6

    Args:
        curves_df: Curves DataFrame (may include forecast-augmented rows from rolling window)
        segment: Target segment
        cohort: Target cohort YYYYMM
        donor_cohort: Donor/template cohort YYYYMM (e.g., '202001', or averaged template)
        mob: Target MOB to forecast
        metric_col: Column name for metric rate (e.g., 'Coll_Principal_Rate')
        mode: 'additive' or 'ratio'
        ref_window: Number of recent actual MOBs used to compute Lc (additive) or
                    the anchor MOB number (ratio)

    Returns:
        dict: Full traceability with all intermediate values:
            - final_rate: Final calculated rate (or None if failed)
            - mode: 'additive' or 'ratio'
            - ref_window: Reference window / anchor MOB used
            - Lc: Target cohort level anchor (mean of actuals)
            - donor_mean_ref: Donor mean over same ref window (additive only)
            - donor_rate_at_mob: Donor rate at forecast MOB
            - shape_adjustment: (rd(m) - r̄d_ref) for additive, scale_factor for ratio
            - success: True if calculation succeeded
            - error: Error message if failed
    """
    cohort_str = clean_cohort(cohort)
    donor_cohort_str = clean_cohort(donor_cohort)

    result = {
        'final_rate': None,
        'mode': mode,
        'ref_window': ref_window,
        'Lc': None,
        'donor_mean_ref': None,
        'donor_rate_at_mob': None,
        'shape_adjustment': None,
        'success': False,
        'error': None
    }

    # -------------------------------------------------------------------------
    # STEP 1: Compute Lc — target cohort's own level anchor from ACTUALS ONLY
    # Filter out forecast-generated rows to prevent rolling contamination.
    # This mirrors the pattern in fn_static_cohort_avg.
    # -------------------------------------------------------------------------
    actuals_mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == cohort_str) &
        (curves_df['MOB'] >= Config.MOB_THRESHOLD)
    )
    # Critical: exclude forecast-generated rows
    if '__is_forecast' in curves_df.columns:
        actuals_mask = actuals_mask & (curves_df['__is_forecast'] != True)

    target_actuals = curves_df[actuals_mask].copy()

    if len(target_actuals) == 0 or metric_col not in target_actuals.columns:
        result['error'] = f"No actual data for target {segment}/{cohort_str}"
        return result

    target_actuals_sorted = target_actuals.sort_values('MOB', ascending=False)

    if mode == 'additive':
        # Lc = mean of most recent ref_window actual MOBs
        recent_actuals = target_actuals_sorted.head(ref_window)
        if len(recent_actuals) < 1:
            result['error'] = f"Insufficient actuals for {cohort_str} (need >=1, got 0)"
            return result
        Lc_values = recent_actuals[metric_col].dropna()
        if len(Lc_values) == 0:
            result['error'] = f"All actuals are NaN for {cohort_str}"
            return result
        Lc = float(Lc_values.mean())
    else:
        # ratio mode: Lc = target rate at single anchor MOB
        # ref_window is treated as the specific anchor MOB number
        anchor_mob = ref_window
        # Find closest available MOB at or before anchor
        available_mobs = target_actuals_sorted['MOB'].values
        valid_anchors = [m for m in available_mobs if m <= anchor_mob]
        if not valid_anchors:
            result['error'] = f"No actuals at or before MOB {anchor_mob} for {cohort_str}"
            return result
        actual_anchor_mob = max(valid_anchors)
        anchor_mask = target_actuals_sorted['MOB'] == actual_anchor_mob
        anchor_row = target_actuals_sorted[anchor_mask]
        if len(anchor_row) == 0:
            result['error'] = f"No target data at anchor MOB {actual_anchor_mob}"
            return result
        Lc = float(anchor_row[metric_col].iloc[0])
        if pd.isna(Lc):
            result['error'] = f"Target rate at anchor MOB {actual_anchor_mob} is NaN"
            return result
        result['ref_window'] = actual_anchor_mob  # Record actual anchor used

    result['Lc'] = Lc

    # -------------------------------------------------------------------------
    # STEP 2: Get donor rate at the forecast MOB
    # -------------------------------------------------------------------------
    donor_mob_mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['Cohort'] == donor_cohort_str) &
        (curves_df['MOB'] == mob)
    )
    donor_mob_data = curves_df[donor_mob_mask]

    if len(donor_mob_data) == 0 or metric_col not in donor_mob_data.columns:
        result['error'] = f"No donor data at forecast MOB {mob} for {donor_cohort_str}"
        return result

    donor_rate_at_mob = donor_mob_data[metric_col].iloc[0]
    if pd.isna(donor_rate_at_mob):
        result['error'] = f"Donor rate at MOB {mob} is NaN"
        return result
    result['donor_rate_at_mob'] = float(donor_rate_at_mob)

    # -------------------------------------------------------------------------
    # STEP 3: Compute shape adjustment and final rate
    # -------------------------------------------------------------------------
    if mode == 'additive':
        # Compute donor mean over the same ref_window MOBs
        # This de-means the donor shape so we're borrowing relative movement only
        donor_ref_mask = (
            (curves_df['Segment'] == segment) &
            (curves_df['Cohort'] == donor_cohort_str) &
            (curves_df['MOB'] >= Config.MOB_THRESHOLD)
        )
        donor_ref_data = curves_df[donor_ref_mask].sort_values('MOB', ascending=False)

        if len(donor_ref_data) == 0:
            result['error'] = f"No donor reference data for {donor_cohort_str}"
            return result

        # Use the same ref_window depth as the target's Lc computation
        donor_ref_window = donor_ref_data.head(ref_window)
        donor_ref_values = donor_ref_window[metric_col].dropna()
        if len(donor_ref_values) == 0:
            result['error'] = f"All donor reference values are NaN for {donor_cohort_str}"
            return result

        donor_mean_ref = float(donor_ref_values.mean())
        result['donor_mean_ref'] = donor_mean_ref

        # Additive shape adjustment: borrow the deviation from donor's own mean
        shape_adjustment = float(donor_rate_at_mob) - donor_mean_ref
        result['shape_adjustment'] = shape_adjustment

        # Final rate = target level + donor shape deviation
        final_rate = Lc + shape_adjustment

    else:  # ratio mode
        # Scale factor = target level / donor rate at anchor MOB
        anchor_mob_ratio = result['ref_window']
        donor_anchor_mask = (
            (curves_df['Segment'] == segment) &
            (curves_df['Cohort'] == donor_cohort_str) &
            (curves_df['MOB'] == anchor_mob_ratio)
        )
        donor_anchor_data = curves_df[donor_anchor_mask]

        if len(donor_anchor_data) == 0:
            result['error'] = f"No donor data at anchor MOB {anchor_mob_ratio} for {donor_cohort_str}"
            return result

        donor_rate_at_anchor = donor_anchor_data[metric_col].iloc[0]
        if pd.isna(donor_rate_at_anchor) or donor_rate_at_anchor == 0:
            result['error'] = f"Donor rate at anchor MOB {anchor_mob_ratio} is 0 or NaN"
            return result

        scale_factor = Lc / float(donor_rate_at_anchor)
        result['shape_adjustment'] = scale_factor  # For ratio mode, shape_adjustment = scale_factor
        result['donor_mean_ref'] = float(donor_rate_at_anchor)  # Reuse field for anchor rate

        final_rate = float(donor_rate_at_mob) * scale_factor

    result['final_rate'] = float(final_rate)
    result['success'] = True

    return result


def fn_seg_median(curves_df: pd.DataFrame, segment: str, mob: int,
                  metric_col: str) -> Optional[float]:
    """
    Median rate across all cohorts in segment at MOB.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        mob: Target MOB
        metric_col: Column name for metric rate

    Returns:
        float or None: Median rate
    """
    # Filter data
    # Filter to genuine historical actuals only:
    # - OpeningGBV > 0 excludes extended flat-lined rows (set to 0.0 in extend_curves)
    # - This also excludes forecast rows added during rolling build (which have no OpeningGBV)
    # Previously used CurveType=='Historical' but that column is not present in working_curves.
    _gbv = curves_df['OpeningGBV'] if 'OpeningGBV' in curves_df.columns else pd.Series(0.0, index=curves_df.index)
    mask = (
        (curves_df['Segment'] == segment) &
        (curves_df['MOB'] == mob) &
        (_gbv > 0)
    )

    data = curves_df[mask]

    if len(data) == 0:
        return None

    if metric_col not in data.columns:
        return None

    rate = data[metric_col].median()

    if pd.isna(rate):
        return None

    return float(rate)


# =============================================================================
# SECTION 9: RATE APPLICATION FUNCTIONS
# =============================================================================

def apply_approach(curves_df: pd.DataFrame, segment: str, cohort: str,
                   mob: int, metric: str, methodology: Dict[str, Any],
                   ds_donor_curves: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculate rate using specified approach.

    Args:
        curves_df: Curves DataFrame
        segment: Target segment
        cohort: Target cohort
        mob: Target MOB
        metric: Target metric
        methodology: Methodology rule dict with Approach, Param1, Param2, Donor_Cohort
        ds_donor_curves: Optional pre-computed DS donor curves DataFrame
            (output of build_ds_donor_curves). Required for DSDonorCRScaled approach.

    Returns:
        dict: Rate and ApproachTag
    """
    approach = methodology['Approach']
    param1 = methodology['Param1']

    # Determine the column name for this metric
    # Some metrics (coverage ratios) don't follow the {metric}_Rate pattern
    if metric in ['Total_Coverage_Ratio', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']:
        metric_col = metric  # These are stored directly without _Rate suffix
    else:
        metric_col = f"{metric}_Rate"

    if approach == 'NoMatch_ERROR':
        return {'Rate': 0.0, 'ApproachTag': 'NoMatch_ERROR'}

    elif approach == 'Zero':
        return {'Rate': 0.0, 'ApproachTag': 'Zero'}

    elif approach == 'Manual':
        try:
            if param1 is None or param1 == 'None' or param1 == 'nan':
                return {'Rate': 0.0, 'ApproachTag': 'Manual_InvalidParam_ERROR'}
            rate = float(param1)
            return {'Rate': rate, 'ApproachTag': 'Manual'}
        except (ValueError, TypeError):
            return {'Rate': 0.0, 'ApproachTag': 'Manual_InvalidParam_ERROR'}

    elif approach == 'CohortAvg':
        try:
            lookback = int(float(param1)) if param1 and param1 != 'None' else Config.LOOKBACK_PERIODS
        except (ValueError, TypeError):
            lookback = Config.LOOKBACK_PERIODS

        # For debt sale metrics, only average non-zero rates
        # (zeros just mean no debt sale occurred that month, not that the rate is 0)
        exclude_zeros = metric in ['WO_DebtSold', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']

        rate = fn_cohort_avg(curves_df, segment, cohort, mob, metric_col, lookback, exclude_zeros)
        if rate is not None:
            tag = 'CohortAvg_NonZero' if exclude_zeros else 'CohortAvg'
            return {'Rate': rate, 'ApproachTag': tag}
        else:
            # Fallback to SegMedian when CohortAvg has insufficient data
            # This prevents newly originated cohorts from getting zero rates
            seg_rate = fn_seg_median(curves_df, segment, mob, metric_col)
            if seg_rate is not None:
                return {'Rate': seg_rate, 'ApproachTag': 'CohortAvg_FallbackSegMedian'}
            else:
                return {'Rate': 0.0, 'ApproachTag': 'CohortAvg_NoData_ERROR'}

    elif approach == 'CohortTrend':
        rate = fn_cohort_trend(curves_df, segment, cohort, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': 'CohortTrend'}
        else:
            # Fallback to SegMedian when CohortTrend has insufficient data
            seg_rate = fn_seg_median(curves_df, segment, mob, metric_col)
            if seg_rate is not None:
                return {'Rate': seg_rate, 'ApproachTag': 'CohortTrend_FallbackSegMedian'}
            else:
                return {'Rate': 0.0, 'ApproachTag': 'CohortTrend_NoData_ERROR'}

    elif approach == 'SegMedian':
        rate = fn_seg_median(curves_df, segment, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': 'SegMedian'}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'SegMedian_NoData_ERROR'}

    elif approach == 'DonorCohort':
        if param1 is None or param1 == 'None':
            return {'Rate': 0.0, 'ApproachTag': 'DonorCohort_NoParam_ERROR'}

        donor = clean_cohort(param1)
        rate = fn_donor_cohort(curves_df, segment, donor, mob, metric_col)
        if rate is not None:
            return {'Rate': rate, 'ApproachTag': f'DonorCohort:{donor}'}
        else:
            return {'Rate': 0.0, 'ApproachTag': f'DonorCohort_NoData_ERROR:{donor}'}

    elif approach == 'ShapeBorrowScaled':
        # ShapeBorrowScaled borrows curve SHAPE from a donor cohort, anchored to the
        # target cohort's own level (actuals-only, no rolling contamination).
        #
        # Param1 = donor cohort (YYYYMM)
        # Param2 = mode:ref_window  e.g. "additive:12" or "ratio:6"
        #          mode only          e.g. "additive"   (defaults: ref_window=12)
        #          omitted            defaults to additive:12
        #
        # ADDITIVE (default): r̂(m) = Lc + (rd(m) - r̄d_ref)
        #   Lc       = mean of target's last ref_window actual rates
        #   r̄d_ref   = mean of donor's last ref_window rates (de-means the shape)
        #   rd(m)    = donor rate at forecast MOB
        #
        # RATIO: r̂(m) = rd(m) × (Lc / rd_anchor)
        #   Lc       = target rate at anchor MOB (ref_window treated as MOB number)
        #   rd_anchor = donor rate at same anchor MOB
        #   rd(m)    = donor rate at forecast MOB

        if param1 is None or param1 == 'None':
            return {'Rate': 0.0, 'ApproachTag': 'ShapeBorrowScaled_NoParam_ERROR'}

        donor = clean_cohort(param1)
        param2 = methodology.get('Param2')

        # Parse mode and ref_window from Param2
        mode = 'additive'
        ref_window = 12
        if param2 and str(param2) not in ('None', 'nan', ''):
            param2_str = str(param2).strip()
            if ':' in param2_str:
                parts = param2_str.split(':', 1)
                mode = parts[0].strip().lower()
                try:
                    ref_window = int(float(parts[1].strip()))
                except (ValueError, TypeError):
                    ref_window = 12
            else:
                # mode only provided, no ref_window specified
                mode = param2_str.lower()
                ref_window = 6 if mode == 'ratio' else 12

        if mode not in ('additive', 'ratio'):
            mode = 'additive'  # safe default

        # Run ShapeBorrowScaled with full traceability
        sbs_result = fn_shape_borrow_scaled(
            curves_df, segment, cohort, donor, mob, metric_col, mode, ref_window
        )

        if sbs_result['success']:
            final_rate = sbs_result['final_rate']
            Lc = sbs_result['Lc']
            adj = sbs_result['shape_adjustment']

            if mode == 'additive':
                tag = f"ShapeBorrowScaled:{donor}(add,Lc={Lc:.4f},adj={adj:+.4f})"
            else:
                tag = f"ShapeBorrowScaled:{donor}(ratio,Lc={Lc:.4f},x{adj:.3f})"

            return {
                'Rate': final_rate,
                'ApproachTag': tag,
                # Traceability columns
                'ShapeBorrow_Donor': donor,
                'ShapeBorrow_Mode': mode,
                'ShapeBorrow_RefWindow': sbs_result['ref_window'],
                'ShapeBorrow_Lc': sbs_result['Lc'],
                'ShapeBorrow_DonorMeanRef': sbs_result['donor_mean_ref'],
                'ShapeBorrow_DonorRateAtMOB': sbs_result['donor_rate_at_mob'],
                'ShapeBorrow_ShapeAdj': sbs_result['shape_adjustment'],
                'ShapeBorrow_FinalRate': final_rate,
            }
        else:
            # Fallback to DonorCohort (raw rate) if ShapeBorrowScaled fails
            rate = fn_donor_cohort(curves_df, segment, donor, mob, metric_col)
            if rate is not None:
                return {
                    'Rate': rate,
                    'ApproachTag': f'ShapeBorrowScaled_FallbackDonor:{donor}',
                    'ShapeBorrow_Error': sbs_result.get('error', 'Unknown error'),
                    'ShapeBorrow_FallbackRate': rate,
                }
            else:
                return {
                    'Rate': 0.0,
                    'ApproachTag': f'ShapeBorrowScaled_NoData_ERROR:{donor}',
                    'ShapeBorrow_Error': sbs_result.get('error', 'Unknown error'),
                }

    elif approach == 'ScaledCohortAvg':
        # ScaledCohortAvg: Same as CohortAvg but applies a scaling factor from Param2
        # Param1 = lookback periods (same as CohortAvg)
        # Param2 = scaling factor (e.g., 1.1 = +10%, 0.9 = -10%)
        try:
            lookback = int(float(param1)) if param1 and param1 != 'None' else Config.LOOKBACK_PERIODS
        except (ValueError, TypeError):
            lookback = Config.LOOKBACK_PERIODS

        # Get scale factor from Param2
        param2 = methodology.get('Param2')
        try:
            scale_factor = float(param2) if param2 and param2 != 'None' and param2 != 'nan' else 1.0
        except (ValueError, TypeError):
            scale_factor = 1.0

        exclude_zeros = metric in ['WO_DebtSold', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']
        rate = fn_cohort_avg(curves_df, segment, cohort, mob, metric_col, lookback, exclude_zeros)

        if rate is not None:
            scaled_rate = rate * scale_factor
            tag = f'ScaledCohortAvg(x{scale_factor:.3f})'
            return {'Rate': scaled_rate, 'ApproachTag': tag}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'ScaledCohortAvg_NoData_ERROR'}

    elif approach == 'StaticCohortAvg':
        # StaticCohortAvg: Cohort average using historical ACTUAL rates only
        # Param1 = lookback periods (default LOOKBACK_PERIODS)
        try:
            lookback = int(float(param1)) if param1 and param1 != 'None' else Config.LOOKBACK_PERIODS
        except (ValueError, TypeError):
            lookback = Config.LOOKBACK_PERIODS

        exclude_zeros = metric in ['WO_DebtSold', 'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']
        rate = fn_static_cohort_avg(curves_df, segment, cohort, mob, metric_col, lookback, exclude_zeros)

        if rate is not None:
            return {'Rate': rate, 'ApproachTag': f'StaticCohortAvg({lookback})'}
        else:
            return {'Rate': 0.0, 'ApproachTag': 'StaticCohortAvg_NoData_ERROR'}

    elif approach == 'DSDonorCRScaled':
        # DSDonorCRScaled: look up the pre-computed donor-derived DS rate from
        # ds_donor_curves (built by build_ds_donor_curves).  The DS rate is
        # non-zero only at the cohort's DS event MOBs.  A segment-level scaling
        # factor (Layer B) is applied separately in run_one_step via the
        # WO_DebtSold_ScaleFactor column in the rate lookup table.
        #
        # Fallback: SegMedian if ds_donor_curves is not available or has no match.
        if ds_donor_curves is not None and len(ds_donor_curves) > 0:
            cohort_clean = clean_cohort(cohort)
            mask = (
                (ds_donor_curves['Segment'] == segment) &
                (ds_donor_curves['Cohort'] == cohort_clean) &
                (ds_donor_curves['MOB'] == mob)
            )
            matches = ds_donor_curves[mask]
            if len(matches) > 0:
                row = matches.iloc[0]
                rate = float(row['WO_DebtSold_Rate_DS'])
                donor = str(row['DonorCohort'])
                auto_flag = '(auto)' if row.get('IsAutoSelected', False) else ''
                return {
                    'Rate': rate,
                    'ApproachTag': f'DSDonorCRScaled:{donor}{auto_flag}',
                }
            # MOB is not a DS event MOB for this cohort → rate is 0 (gated in run_one_step)
            return {'Rate': 0.0, 'ApproachTag': 'DSDonorCRScaled:NonEventMOB'}

        # Fallback if ds_donor_curves not yet built
        seg_rate = fn_seg_median(curves_df, segment, mob, metric_col)
        if seg_rate is not None:
            return {'Rate': seg_rate, 'ApproachTag': 'DSDonorCRScaled_FallbackSegMedian'}
        return {'Rate': 0.0, 'ApproachTag': 'DSDonorCRScaled_NoData_ERROR'}

    else:
        return {'Rate': 0.0, 'ApproachTag': f'UnknownApproach_ERROR:{approach}'}


def apply_rate_cap(rate: float, metric: str, approach_tag: str) -> float:
    """
    Cap rates to reasonable ranges.

    Args:
        rate: Input rate
        metric: Metric name
        approach_tag: Approach tag (caps bypassed for Manual and ERROR)

    Returns:
        float: Capped rate
    """
    if rate is None or pd.isna(rate):
        return 0.0

    # Don't cap Manual overrides or errors
    if 'Manual' in approach_tag or 'ERROR' in approach_tag:
        return rate

    # Apply caps
    if metric in Config.RATE_CAPS:
        min_cap, max_cap = Config.RATE_CAPS[metric]
        return max(min_cap, min(max_cap, rate))

    return rate


# =============================================================================
# SECTION 9B: DEBT SALE FORECASTING FUNCTIONS (DSDonorCRScaled)
# =============================================================================
#
# Implements a hierarchical DS write-off forecast replacing the SegMedian approach.
#
# Layer A – Cohort-level DS curve generation (shape / allocation):
#   For each cohort, build a WO_DebtSold_Rate curve indexed by MOB using a donor
#   cohort selected via CR-curve similarity (or manually specified in the
#   Rate_Methodology CSV via the Donor_Cohort column).  A shrinkage formula blends
#   the cohort's own sparse history toward the donor curve.
#
# Layer B – Segment-level scaling (volume control):
#   A per-segment scaling factor is calibrated on "normal" DS quarters
#   (the most recent N non-outlier DS event quarters, excluding DS_OUTLIER_MONTHS).
#   Applying this factor ensures cohort-level sums match historical quarterly behaviour.
#
# Both the raw (pre-scale) and scaled values are written to the forecast output.
# The scaled value is used in the GBV bridge and impairment calculation.
# =============================================================================


def _get_ds_event_mobs_for_cohort(cohort: str, max_mob: int = 120) -> List[int]:
    """
    Return the list of MOBs that are DS event months for a cohort.

    A DS event month is one whose calendar month falls in Config.DS_MONTHS.
    The origination month is inferred from the last two digits of the cohort string
    (YYYYMM format).

    Args:
        cohort: Cohort string in YYYYMM format (e.g. '202201')
        max_mob: Maximum MOB to consider

    Returns:
        List[int]: MOBs (1-indexed) that align with DS calendar months
    """
    cohort_clean = clean_cohort(cohort)
    try:
        orig_month = int(cohort_clean[-2:])
    except (ValueError, IndexError):
        orig_month = 1  # fallback to January

    ds_event_months = set(Config.DS_MONTHS)
    return [
        mob for mob in range(1, max_mob + 1)
        if (((orig_month - 1) + (mob - 1)) % 12 + 1) in ds_event_months
    ]


def build_ds_event_history(fact_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean DS event-month history from the raw actuals.

    Filters to DS calendar months only (Config.DS_MONTHS) and excludes any
    (year, month) pairs listed in Config.DS_OUTLIER_MONTHS (e.g. Dec-25).
    Computes the event-month rate WO_DebtSold_Rate_Event = WO_DebtSold / OpeningGBV.

    Args:
        fact_raw: Raw historical data DataFrame (must include CalendarMonth, Segment,
                  Cohort, MOB, OpeningGBV, WO_DebtSold)

    Returns:
        pd.DataFrame: DS event rows with columns
            Segment, Cohort, MOB, CalendarMonth, OpeningGBV, WO_DebtSold,
            WO_DebtSold_Rate_Event
    """
    df = fact_raw.copy()

    # Keep only DS calendar months
    df = df[df['CalendarMonth'].dt.month.isin(Config.DS_MONTHS)].copy()

    # Remove outlier months (e.g. Dec-25)
    for (yr, mo) in Config.DS_OUTLIER_MONTHS:
        df = df[~((df['CalendarMonth'].dt.year == yr) &
                  (df['CalendarMonth'].dt.month == mo))]

    # Keep only rows with positive OpeningGBV
    df = df[df['OpeningGBV'] > 0].copy()

    df['WO_DebtSold_Rate_Event'] = df.apply(
        lambda r: safe_divide(r['WO_DebtSold'], r['OpeningGBV']), axis=1
    )

    cols = ['Segment', 'Cohort', 'MOB', 'CalendarMonth',
            'OpeningGBV', 'WO_DebtSold', 'WO_DebtSold_Rate_Event']
    df = df[[c for c in cols if c in df.columns]].copy()

    logger.info(
        f"DS event history: {len(df)} event-month rows across "
        f"{df['Segment'].nunique()} segments "
        f"({df['CalendarMonth'].min().strftime('%Y-%m') if len(df) else 'n/a'} – "
        f"{df['CalendarMonth'].max().strftime('%Y-%m') if len(df) else 'n/a'})"
    )
    return df.reset_index(drop=True)


def _find_auto_donor(
    ds_event_df: pd.DataFrame,
    curves_base_df: pd.DataFrame,
    segment: str,
    target_cohort: str,
) -> Optional[str]:
    """
    Auto-select a donor cohort for the target using CR-profile similarity.

    Computes correlation distance (1 – Pearson correlation) between the target's
    Total_Coverage_Ratio profile and each eligible donor's profile over
    Config.CR_SIM_MOB_START … Config.CR_SIM_MOB_END.

    A donor must have >= Config.DS_MIN_OBS_DONOR DS event observations
    (in ds_event_df) to be eligible.

    Args:
        ds_event_df: DS event history (from build_ds_event_history)
        curves_base_df: Historical rate curves (Segment × Cohort × MOB)
        segment: Segment name
        target_cohort: Target cohort string

    Returns:
        str or None: Best matching donor cohort, or None if no eligible donors found
    """
    # Eligible donors: >= DS_MIN_OBS_DONOR DS events, must not be the target itself
    obs_counts = (
        ds_event_df[ds_event_df['Segment'] == segment]
        .groupby('Cohort')
        .size()
    )
    eligible = [
        c for c, n in obs_counts.items()
        if n >= Config.DS_MIN_OBS_DONOR and c != clean_cohort(target_cohort)
    ]
    if not eligible:
        return None

    mob_range = list(range(Config.CR_SIM_MOB_START, Config.CR_SIM_MOB_END + 1))
    seg_df = curves_base_df[curves_base_df['Segment'] == segment]

    def get_cr_profile(cohort: str) -> Optional[np.ndarray]:
        rows = (
            seg_df[seg_df['Cohort'] == cohort]
            .set_index('MOB')['Total_Coverage_Ratio']
            .reindex(mob_range)
        )
        vals = rows.values
        if np.isnan(vals).all():
            return None
        # Forward-fill then backward-fill NaNs within the window
        s = pd.Series(vals).ffill().bfill()
        return s.values

    target_profile = get_cr_profile(clean_cohort(target_cohort))
    if target_profile is None or target_profile.std() == 0:
        # Cannot compute correlation – fall back to first eligible donor
        return eligible[0]

    best_donor: Optional[str] = None
    best_distance = float('inf')

    for donor in eligible:
        donor_profile = get_cr_profile(donor)
        if donor_profile is None:
            continue
        # Use only positions where both profiles have data
        valid = ~(np.isnan(target_profile) | np.isnan(donor_profile))
        if valid.sum() < 3:
            continue
        tv = target_profile[valid]
        dv = donor_profile[valid]
        if dv.std() == 0:
            continue
        corr = np.corrcoef(tv, dv)[0, 1]
        distance = 1.0 - corr
        if distance < best_distance:
            best_distance = distance
            best_donor = donor

    return best_donor


def build_ds_donor_curves(
    ds_event_df: pd.DataFrame,
    methodology_df: pd.DataFrame,
    curves_base_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-cohort DS rate curves (Layer A).

    For each (Segment, Cohort) present in curves_base_df:
    1. Find the donor cohort:
       - If Rate_Methodology has a DSDonorCRScaled row with a non-AUTO Donor_Cohort,
         use that cohort.
       - Otherwise (AUTO / not specified), run CR-similarity to find the best donor.
    2. Align DS events by event-number (not strict MOB) so that origination-month
       differences between target and donor do not cause zero rates.
    3. Apply shrinkage:  alpha = min(1, N_own / DS_SHRINKAGE_TARGET)
       blended_rate = alpha × own_rate + (1-alpha) × donor_rate
    4. Apply per-segment DS rate cap.
    5. Extend beyond the donor's known history by holding the last observed
       donor event rate flat.

    Returns one row per (Segment, Cohort, MOB) for every MOB that is a DS event
    month for that cohort (up to 120 MOBs).  Non-DS-event MOBs are omitted
    (the rate lookup returns 0.0 for those via the DSDonorCRScaled handler).

    Args:
        ds_event_df: Output of build_ds_event_history
        methodology_df: Loaded Rate_Methodology DataFrame (may have Donor_Cohort col)
        curves_base_df: Historical rate curves (Segment × Cohort × MOB)

    Returns:
        pd.DataFrame: Columns:
            Segment, Cohort, MOB, WO_DebtSold_Rate_DS, DonorCohort,
            IsAutoSelected, N_DS_Obs_Target, N_DS_Obs_Donor, ShrinkageAlpha,
            DS_Event_Number
    """
    # Pre-compute per-segment DS rate caps from historical maxima
    seg_caps: Dict[str, float] = {}
    for seg in ds_event_df['Segment'].unique():
        seg_max = ds_event_df[ds_event_df['Segment'] == seg]['WO_DebtSold_Rate_Event'].max()
        seg_caps[seg] = float(seg_max) * Config.DS_RATE_CAP_MULTIPLIER if seg_max > 0 else 1.0

    results = []

    all_segments = curves_base_df['Segment'].unique()
    for seg in all_segments:
        seg_df = curves_base_df[curves_base_df['Segment'] == seg]
        cohorts = seg_df['Cohort'].unique()
        ds_cap = seg_caps.get(seg, 1.0)

        # Build a lookup: cohort → its DS event rates by event number (0-based).
        # We aggregate to CalendarMonth level first so that clustered cohorts
        # (which have many MOBs per CalendarMonth in the raw data) are treated
        # as one DS event per quarter rather than one per MOB.
        def cohort_ds_events(coh: str) -> pd.Series:
            """Returns CalendarMonth-aggregated DS event rates (0-based event index)."""
            rows = ds_event_df[
                (ds_event_df['Segment'] == seg) &
                (ds_event_df['Cohort'] == coh)
            ]
            if len(rows) == 0:
                return pd.Series(dtype=float)
            # Aggregate WO_DebtSold and OpeningGBV across all MOBs per CalendarMonth
            agg = (
                rows.groupby('CalendarMonth')
                .agg(WO_DebtSold=('WO_DebtSold', 'sum'),
                     OpeningGBV=('OpeningGBV', 'sum'))
                .reset_index()
                .sort_values('CalendarMonth')
            )
            agg['rate'] = agg['WO_DebtSold'] / agg['OpeningGBV'].clip(lower=1e-8)
            return agg['rate'].reset_index(drop=True)

        for cohort in cohorts:
            cohort_clean = clean_cohort(cohort)

            # --- 1. Find donor ---
            meth = get_methodology(methodology_df, seg, cohort_clean, 0, 'WO_DebtSold')
            donor_cohort_raw = meth.get('Donor_Cohort')
            is_auto = False

            if (meth['Approach'] == 'DSDonorCRScaled'
                    and donor_cohort_raw
                    and str(donor_cohort_raw).upper() not in ('AUTO', 'NONE', 'NAN', '')):
                donor_cohort = clean_cohort(str(donor_cohort_raw))
            else:
                # Auto-select via CR similarity
                donor_cohort = _find_auto_donor(ds_event_df, curves_base_df, seg, cohort_clean)
                is_auto = True

            # --- 2. Gather own and donor event-rate series ---
            own_events = cohort_ds_events(cohort_clean)
            n_own = len(own_events)

            donor_events: pd.Series
            n_donor = 0
            if donor_cohort:
                donor_events = cohort_ds_events(donor_cohort)
                n_donor = len(donor_events)
            else:
                # No donor available: use segment-level median rate as flat donor curve
                seg_median_rate = ds_event_df[
                    ds_event_df['Segment'] == seg
                ]['WO_DebtSold_Rate_Event'].median()
                donor_events = pd.Series(
                    [seg_median_rate if pd.notna(seg_median_rate) else 0.0] * max(1, n_own)
                )
                donor_cohort = '_SegMedian'
                is_auto = True

            # --- 3. Shrinkage weight ---
            alpha = min(1.0, n_own / Config.DS_SHRINKAGE_TARGET) if Config.DS_SHRINKAGE_TARGET > 0 else 1.0

            # --- 4. DS event MOBs for this target cohort ---
            target_ds_mobs = _get_ds_event_mobs_for_cohort(cohort_clean, max_mob=120)

            # Last observed donor event rate (for forward extension beyond donor history)
            last_donor_rate = float(donor_events.iloc[-1]) if len(donor_events) > 0 else 0.0

            for event_num, target_mob in enumerate(target_ds_mobs):
                # Donor rate for this event number (hold last flat if beyond donor history)
                if event_num < len(donor_events):
                    donor_rate = float(donor_events.iloc[event_num])
                    last_donor_rate = donor_rate
                else:
                    donor_rate = last_donor_rate  # tail extension

                if event_num < n_own:
                    # Own data exists for this event: blend with shrinkage weight
                    own_rate = float(own_events.iloc[event_num])
                    blended = alpha * own_rate + (1.0 - alpha) * donor_rate
                else:
                    # No own data yet for this future event: use 100% donor rate.
                    # Applying alpha × 0 + (1-alpha) × donor would wrongly give only
                    # (1-alpha) fraction of the donor rate for cohorts with partial history.
                    blended = donor_rate

                # Apply cap
                blended = min(blended, ds_cap)

                results.append({
                    'Segment': seg,
                    'Cohort': cohort_clean,
                    'MOB': target_mob,
                    'WO_DebtSold_Rate_DS': blended,
                    'DonorCohort': donor_cohort,
                    'IsAutoSelected': is_auto,
                    'N_DS_Obs_Target': n_own,
                    'N_DS_Obs_Donor': n_donor,
                    'ShrinkageAlpha': round(alpha, 4),
                    'DS_Event_Number': event_num + 1,
                })

    if not results:
        logger.warning("build_ds_donor_curves: no rows produced")
        return pd.DataFrame(columns=[
            'Segment', 'Cohort', 'MOB', 'WO_DebtSold_Rate_DS',
            'DonorCohort', 'IsAutoSelected', 'N_DS_Obs_Target',
            'N_DS_Obs_Donor', 'ShrinkageAlpha', 'DS_Event_Number',
        ])

    df_out = pd.DataFrame(results)
    logger.info(
        f"build_ds_donor_curves: {len(df_out)} rows across "
        f"{df_out['Segment'].nunique()} segments, "
        f"{df_out['Cohort'].nunique()} cohorts"
    )
    return df_out


def calibrate_ds_scaling(
    fact_raw: pd.DataFrame,
    ds_event_df: pd.DataFrame,
    ds_donor_curves_df: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Calibrate per-segment Layer B scaling factors on normal DS quarters (Layer B).

    Calibration months = the most recent Config.DS_CALIBRATION_N_QUARTERS non-outlier
    DS event quarters found in fact_raw (typically Mar, Jun, Sep-25; Dec-25 is already
    excluded by build_ds_event_history / DS_OUTLIER_MONTHS).

    For each calibration month m and segment s:
        actual_m  = Σ WO_DebtSold (actuals for that segment/month)
        implied_m = Σ OpeningGBV × WO_DebtSold_Rate_DS  (from ds_donor_curves_df)
        ratio_m   = actual_m / implied_m  (winsorised to [0.1, 10] and set to 1.0
                                           with a warning if implied is 0)

    ScaleFactor_s = median(ratio_m) over calibration months.

    Args:
        fact_raw: Full raw actuals
        ds_event_df: DS event history (already filtered / outlier-excluded)
        ds_donor_curves_df: Output of build_ds_donor_curves

    Returns:
        Tuple:
          - Dict[str, float]: segment → ScaleFactor (1.0 if calibration not possible)
          - pd.DataFrame: diagnostic table with per-month ratios and final factors
    """
    # Determine calibration months: last N unique CalendarMonths in ds_event_df
    cal_months_all = sorted(ds_event_df['CalendarMonth'].unique(), reverse=True)
    cal_months = cal_months_all[:Config.DS_CALIBRATION_N_QUARTERS]

    if not cal_months:
        logger.warning("calibrate_ds_scaling: no calibration months found")
        segments = fact_raw['Segment'].unique() if 'Segment' in fact_raw.columns else []
        return {s: 1.0 for s in segments}, pd.DataFrame()

    logger.info(
        f"DS scaling calibration months: "
        + ", ".join(m.strftime('%Y-%m') for m in sorted(cal_months))
    )

    diag_rows = []
    seg_ratios: Dict[str, List[float]] = {}

    # Build (Segment, Cohort, CalendarMonth) → event_number mapping.
    # This handles clustered cohorts (many MOBs per CalendarMonth) by
    # aggregating to CalendarMonth level before assigning event numbers.
    ds_agg_hist = (
        ds_event_df
        .groupby(['Segment', 'Cohort', 'CalendarMonth'])
        .agg(WO_DebtSold=('WO_DebtSold', 'sum'),
             OpeningGBV=('OpeningGBV', 'sum'))
        .reset_index()
        .sort_values(['Segment', 'Cohort', 'CalendarMonth'])
    )
    cal_event_num: Dict[tuple, int] = {}
    for (seg_c, coh_c), grp in ds_agg_hist.groupby(['Segment', 'Cohort']):
        for ev_num, (_, row) in enumerate(grp.iterrows(), start=1):
            cal_event_num[(seg_c, clean_cohort(str(coh_c)), row['CalendarMonth'])] = ev_num

    # Build (Segment, Cohort, DS_Event_Number) → rate lookup from donor curves
    if len(ds_donor_curves_df) > 0:
        ds_curve_by_event = ds_donor_curves_df.set_index(
            ['Segment', 'Cohort', 'DS_Event_Number']
        )['WO_DebtSold_Rate_DS']
    else:
        ds_curve_by_event = pd.Series(dtype=float)

    for cal_month in cal_months:
        # Actual DS totals for this month
        actual_month = fact_raw[fact_raw['CalendarMonth'] == cal_month]

        for seg in fact_raw['Segment'].unique():
            actual_seg = actual_month[actual_month['Segment'] == seg]['WO_DebtSold'].sum()

            # Implied DS: aggregate OpeningGBV by cohort (summing all MOBs),
            # then look up the donor curve rate via the event-number mapping.
            # This correctly handles clustered cohorts whose GBV is spread
            # across many MOBs per CalendarMonth.
            seg_month = actual_month[
                (actual_month['Segment'] == seg) &
                (actual_month['OpeningGBV'] > 0)
            ]
            cohort_gbv = seg_month.groupby('Cohort')['OpeningGBV'].sum()

            implied_seg = 0.0
            for coh, gbv in cohort_gbv.items():
                coh_clean = clean_cohort(str(coh))
                ev_key = (seg, coh_clean, cal_month)
                event_num = cal_event_num.get(ev_key)
                if event_num is not None:
                    rate_key = (seg, coh_clean, event_num)
                    if rate_key in ds_curve_by_event.index:
                        rate = float(ds_curve_by_event[rate_key])
                    else:
                        rate = 0.0
                    implied_seg += gbv * rate

            # Compute ratio
            note = ''
            if implied_seg <= 0:
                ratio = 1.0
                note = 'Implied=0; ratio set to 1'
                logger.warning(
                    f"DS scaling: implied=0 for {seg} in "
                    f"{cal_month.strftime('%Y-%m')} – setting ratio=1"
                )
            else:
                ratio = actual_seg / implied_seg
                # Winsorise to [0.1, 10]
                if ratio < 0.1 or ratio > 10:
                    note = f'Winsorised from {ratio:.3f}'
                    ratio = max(0.1, min(10.0, ratio))

            seg_ratios.setdefault(seg, []).append(ratio)
            diag_rows.append({
                'Segment': seg,
                'CalibrationMonth': cal_month.strftime('%Y-%m'),
                'Actual_DS': round(actual_seg, 2),
                'Implied_DS_PreScale': round(implied_seg, 2),
                'Ratio': round(ratio, 4),
                'Note': note,
            })

    # Final scale factor per segment = median of calibration-month ratios
    scale_factors: Dict[str, float] = {}
    for seg, ratios in seg_ratios.items():
        sf = float(np.median(ratios))
        scale_factors[seg] = sf

    # Annotate diagnostic table with final factor
    sf_map = scale_factors
    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df) > 0:
        diag_df['FinalScaleFactor'] = diag_df['Segment'].map(sf_map).fillna(1.0)
        diag_df['CalibrationMonths'] = ', '.join(
            m.strftime('%Y-%m') for m in sorted(cal_months)
        )

    for seg, sf in scale_factors.items():
        logger.info(f"  DS scale factor [{seg}]: {sf:.4f}")

    return scale_factors, diag_df


def compute_cr_similarity_map(curves_base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline diagnostic tool: compute full CR-similarity matrix for all cohorts
    within each segment and return suggested donor mappings.

    This function is NOT called during normal forecast runs.  Run it separately
    (e.g., from a Jupyter notebook or a CLI wrapper) to review donor suggestions
    before populating Donor_Cohort in Rate_Methodology.csv.

    Args:
        curves_base_df: Historical rate curves (Segment × Cohort × MOB)

    Returns:
        pd.DataFrame: Columns:
            Segment, Cohort, SuggestedDonor, CorrelationDistance,
            CR_MOB_Start, CR_MOB_End
    """
    mob_range = list(range(Config.CR_SIM_MOB_START, Config.CR_SIM_MOB_END + 1))
    results = []

    for seg in curves_base_df['Segment'].unique():
        seg_df = curves_base_df[curves_base_df['Segment'] == seg]
        cohorts = seg_df['Cohort'].unique()

        # Build profile matrix
        profiles: Dict[str, np.ndarray] = {}
        for coh in cohorts:
            vals = (
                seg_df[seg_df['Cohort'] == coh]
                .set_index('MOB')['Total_Coverage_Ratio']
                .reindex(mob_range)
            )
            s = pd.Series(vals.values).ffill().bfill()
            profiles[coh] = s.values

        for target in cohorts:
            tv = profiles[target]
            if np.isnan(tv).all() or tv.std() == 0:
                results.append({
                    'Segment': seg, 'Cohort': target,
                    'SuggestedDonor': None, 'CorrelationDistance': None,
                    'CR_MOB_Start': Config.CR_SIM_MOB_START,
                    'CR_MOB_End': Config.CR_SIM_MOB_END,
                })
                continue

            best_donor = None
            best_dist = float('inf')
            for donor in cohorts:
                if donor == target:
                    continue
                dv = profiles[donor]
                valid = ~(np.isnan(tv) | np.isnan(dv))
                if valid.sum() < 3:
                    continue
                if dv[valid].std() == 0:
                    continue
                corr = np.corrcoef(tv[valid], dv[valid])[0, 1]
                dist = 1.0 - corr
                if dist < best_dist:
                    best_dist = dist
                    best_donor = donor

            results.append({
                'Segment': seg,
                'Cohort': target,
                'SuggestedDonor': best_donor,
                'CorrelationDistance': round(best_dist, 4) if best_donor else None,
                'CR_MOB_Start': Config.CR_SIM_MOB_START,
                'CR_MOB_End': Config.CR_SIM_MOB_END,
            })

    return pd.DataFrame(results)


# =============================================================================
# SECTION 10: RATE LOOKUP BUILDER
# =============================================================================

def build_rate_lookup(seed: pd.DataFrame, curves: pd.DataFrame,
                      methodology: pd.DataFrame, max_months: int,
                      ds_donor_curves: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build rate lookup table for forecast with rolling CohortAvg.

    For CohortAvg approach, forecasted rates from month N feed into month N+1's
    calculation. This creates a rolling average where the lookback window includes
    previously forecasted values.

    Args:
        seed: Seed curves
        curves: Extended curves
        methodology: Methodology rules
        max_months: Forecast horizon
        ds_donor_curves: Optional pre-computed DS donor curves (from build_ds_donor_curves).
            When provided, enables the DSDonorCRScaled approach for WO_DebtSold.

    Returns:
        pd.DataFrame: Rate lookup table
    """
    logger.info("Building rate lookup...")

    # Metrics to calculate rates for
    rate_metrics = [
        'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other', 'ContraSettlements_Principal',
        'ContraSettlements_Interest', 'NewLoanAmount'
    ]

    # Create a working copy of curves that we'll update with forecasted rates
    working_curves = curves.copy()
    if '__is_forecast' not in working_curves.columns:
        working_curves['__is_forecast'] = False

    lookups = []

    # Process month-by-month to enable rolling CohortAvg
    # This ensures month N+1's CohortAvg includes month N's forecasted rate
    for month_offset in range(max_months):
        month_forecasts = []

        for _, seed_row in seed.iterrows():
            segment = seed_row['Segment']
            cohort = seed_row['Cohort']
            start_mob = seed_row['MOB']
            mob = start_mob + month_offset

            row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
            }

            forecast_rates = {}  # Store rates to add to working_curves

            for metric in rate_metrics:
                metric_col = f'{metric}_Rate'

                # Get methodology
                meth = get_methodology(methodology, segment, cohort, mob, metric)

                # Apply approach using working_curves (includes previous forecasts)
                result = apply_approach(
                    working_curves, segment, cohort, mob, metric, meth,
                    ds_donor_curves=ds_donor_curves,
                )

                # Apply cap
                capped_rate = apply_rate_cap(result['Rate'], metric, result['ApproachTag'])

                row[f'{metric}_Rate'] = capped_rate
                row[f'{metric}_Approach'] = result['ApproachTag']

                # Store rate for adding to working_curves
                forecast_rates[metric_col] = capped_rate

            lookups.append(row)

            # Store this forecast to add to working_curves after processing all cohorts
            month_forecasts.append({
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
                '__is_forecast': True,
                **forecast_rates
            })

        # After processing all cohorts for this month, add forecasted rates to working_curves
        # This enables rolling CohortAvg for subsequent months
        if month_forecasts:
            forecast_df = pd.DataFrame(month_forecasts)
            working_curves = pd.concat([working_curves, forecast_df], ignore_index=True)

    lookup_df = pd.DataFrame(lookups)
    logger.info(f"Built rate lookup with {len(lookup_df)} entries")

    return lookup_df


def build_impairment_lookup(seed: pd.DataFrame, impairment_curves: pd.DataFrame,
                            methodology: pd.DataFrame, max_months: int,
                            debt_sale_schedule: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build impairment lookup table for forecast with rolling CohortAvg.

    For CohortAvg approach on Total_Coverage_Ratio, forecasted ratios from month N
    feed into month N+1's calculation. This creates a rolling average where the
    lookback window includes previously forecasted values.

    Args:
        seed: Seed curves
        impairment_curves: Impairment curves
        methodology: Methodology rules
        max_months: Forecast horizon
        debt_sale_schedule: Optional debt sale schedule

    Returns:
        pd.DataFrame: Impairment lookup table
    """
    logger.info("Building impairment lookup...")

    # Get start forecast month from seed
    start_forecast_month = seed['ForecastMonth'].iloc[0]

    # Create a working copy of impairment_curves that we'll update with forecasted ratios
    working_curves = impairment_curves.copy()

    lookups = []

    # Process month-by-month to enable rolling CohortAvg
    for month_offset in range(max_months):
        forecast_month = end_of_month(start_forecast_month + relativedelta(months=month_offset))
        month_forecasts = []

        for _, seed_row in seed.iterrows():
            segment = seed_row['Segment']
            cohort = seed_row['Cohort']
            start_mob = seed_row['MOB']
            mob = start_mob + month_offset

            row = {
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
                'ForecastMonth': forecast_month,
            }

            # Check if this is a debt sale month
            debt_sale_wo = 0.0
            if debt_sale_schedule is not None:
                ds_mask = (
                    (debt_sale_schedule['ForecastMonth'] == forecast_month) &
                    (debt_sale_schedule['Segment'] == segment) &
                    (debt_sale_schedule['Cohort'] == cohort)
                )
                if ds_mask.any():
                    ds_row = debt_sale_schedule[ds_mask].iloc[0]
                    debt_sale_wo = ds_row.get('Debt_Sale_WriteOffs', 0.0)
                    row['Debt_Sale_Coverage_Ratio'] = ds_row.get('Debt_Sale_Coverage_Ratio', 0.85)
                    row['Debt_Sale_Proceeds_Rate'] = ds_row.get('Debt_Sale_Proceeds_Rate', 0.90)

            row['Debt_Sale_WriteOffs'] = debt_sale_wo

            # Get coverage ratio from methodology using working_curves (includes previous forecasts)
            meth = get_methodology(methodology, segment, cohort, mob, 'Total_Coverage_Ratio')
            result = apply_approach(working_curves, segment, cohort, mob, 'Total_Coverage_Ratio', meth)

            if result['Rate'] == 0.0 and 'ERROR' in result['ApproachTag']:
                # Fallback to curves if available
                mask = (
                    (working_curves['Segment'] == segment) &
                    (working_curves['Cohort'] == cohort)
                )
                if mask.any():
                    avg_coverage = working_curves[mask]['Total_Coverage_Ratio'].mean()
                    if not pd.isna(avg_coverage):
                        result['Rate'] = avg_coverage

            capped_rate = apply_rate_cap(result['Rate'], 'Total_Coverage_Ratio', result['ApproachTag'])

            # Apply seasonal adjustment if enabled
            # The base rate from approaches is considered "de-seasonalized"
            # We re-apply seasonality based on the forecast month
            if Config.ENABLE_SEASONALITY:
                forecast_month_num = forecast_month.month
                seasonal_factor = get_seasonal_factor(segment, forecast_month_num)
                final_rate = capped_rate * seasonal_factor
                approach_tag = f"{result['ApproachTag']}+Seasonal({seasonal_factor:.3f})"
                row['Total_Coverage_Ratio_Base'] = capped_rate  # Store base rate for transparency
                row['Seasonal_Factor'] = seasonal_factor
            else:
                final_rate = capped_rate
                approach_tag = result['ApproachTag']
                row['Total_Coverage_Ratio_Base'] = capped_rate
                row['Seasonal_Factor'] = 1.0

            row['Total_Coverage_Ratio'] = final_rate
            row['Total_Coverage_Approach'] = approach_tag

            # Copy ShapeBorrowScaled traceability columns if present
            for key in ['ShapeBorrow_Donor', 'ShapeBorrow_Mode', 'ShapeBorrow_RefWindow',
                        'ShapeBorrow_Lc', 'ShapeBorrow_DonorMeanRef',
                        'ShapeBorrow_DonorRateAtMOB', 'ShapeBorrow_ShapeAdj',
                        'ShapeBorrow_FinalRate', 'ShapeBorrow_Error', 'ShapeBorrow_FallbackRate']:
                if key in result:
                    row[key] = result[key]

            # Set defaults for debt sale ratios if not already set
            if 'Debt_Sale_Coverage_Ratio' not in row:
                row['Debt_Sale_Coverage_Ratio'] = 0.85
            if 'Debt_Sale_Proceeds_Rate' not in row:
                row['Debt_Sale_Proceeds_Rate'] = 0.90

            lookups.append(row)

            # Store this forecast to add to working_curves after processing all cohorts
            month_forecasts.append({
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob,
                'Total_Coverage_Ratio': capped_rate
            })

        # After processing all cohorts for this month, add forecasted ratios to working_curves
        # This enables rolling CohortAvg for subsequent months
        if month_forecasts:
            forecast_df = pd.DataFrame(month_forecasts)
            working_curves = pd.concat([working_curves, forecast_df], ignore_index=True)

    lookup_df = pd.DataFrame(lookups)
    logger.info(f"Built impairment lookup with {len(lookup_df)} entries")

    return lookup_df


# =============================================================================
# SECTION 11: FORECAST ENGINE FUNCTIONS
# =============================================================================

def run_one_step(seed_table: pd.DataFrame, rate_lookup: pd.DataFrame,
                 impairment_lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute one month of forecast.

    Args:
        seed_table: Current seed with BoM, MOB, ForecastMonth
        rate_lookup: Rate lookup table
        impairment_lookup: Impairment lookup table

    Returns:
        tuple: (step_output_df, next_seed_df)
    """
    outputs = []
    next_seeds = []

    for _, seed_row in seed_table.iterrows():
        segment = seed_row['Segment']
        cohort = seed_row['Cohort']
        mob = seed_row['MOB']
        bom = seed_row['BoM']
        forecast_month = seed_row['ForecastMonth']
        prior_provision = seed_row.get('Prior_Provision_Balance', 0.0)

        # Get rates
        rate_mask = (
            (rate_lookup['Segment'] == segment) &
            (rate_lookup['Cohort'] == cohort) &
            (rate_lookup['MOB'] == mob)
        )

        if not rate_mask.any():
            continue

        rates = rate_lookup[rate_mask].iloc[0]

        # Get impairment rates
        imp_mask = (
            (impairment_lookup['Segment'] == segment) &
            (impairment_lookup['Cohort'] == cohort) &
            (impairment_lookup['MOB'] == mob)
        )

        if not imp_mask.any():
            continue

        imp_rates = impairment_lookup[imp_mask].iloc[0]

        # Calculate amounts
        opening_gbv = bom

        new_loan_amount = opening_gbv * rates.get('NewLoanAmount_Rate', 0.0)
        coll_principal = opening_gbv * rates.get('Coll_Principal_Rate', 0.0)
        coll_interest = opening_gbv * rates.get('Coll_Interest_Rate', 0.0)
        interest_revenue = opening_gbv * rates.get('InterestRevenue_Rate', 0.0) / 12  # Monthly

        # WO_DebtSold only occurs in debt sale months (Mar, Jun, Sep, Dec)
        # For DSDonorCRScaled: compute raw amount (pre-scale) then apply Layer B factor.
        # For all other approaches: scale_factor = 1.0 (no change).
        if is_debt_sale_month(forecast_month):
            wo_debt_sold_raw = opening_gbv * rates.get('WO_DebtSold_Rate', 0.0)
            ds_scale_factor = float(rates.get('WO_DebtSold_ScaleFactor', 1.0))
            wo_debt_sold = wo_debt_sold_raw * ds_scale_factor  # scaled value used in GBV
        else:
            wo_debt_sold_raw = 0.0
            ds_scale_factor = float(rates.get('WO_DebtSold_ScaleFactor', 1.0))
            wo_debt_sold = 0.0

        wo_other = opening_gbv * rates.get('WO_Other_Rate', 0.0)
        contra_principal = opening_gbv * rates.get('ContraSettlements_Principal_Rate', 0.0)
        contra_interest = opening_gbv * rates.get('ContraSettlements_Interest_Rate', 0.0)

        # Calculate closing GBV (uses scaled WO_DebtSold so GBV bridge is consistent)
        closing_gbv = (
            opening_gbv +
            interest_revenue -
            abs(coll_principal) -
            abs(coll_interest) -
            wo_debt_sold -
            wo_other
        )

        # Ensure non-negative
        closing_gbv = max(0.0, closing_gbv)

        # =======================================================================
        # DEBT SALE AND IMPAIRMENT CALCULATION
        # =======================================================================
        # Calculation flow per user specification:
        # 1. Total provision balance = Closing GBV × Coverage Ratio
        # 2. Total provision movement = Provision[t] - Provision[t-1]
        # 3. DS provision release = DS Coverage Ratio × DS WriteOffs (sale months only)
        # 4. DS proceeds = DS Proceeds Rate × DS WriteOffs (sale months only)
        # 5. Non-DS provision movement = Total provision movement + DS provision release
        # 6. Gross impairment (excl DS) = Non-DS provision movement + WO_Other
        # 7. Debt sale impact = DS WriteOffs + DS provision release + DS proceeds
        # 8. Net impairment = Gross impairment (excl DS) + Debt sale impact
        #
        # SIGN CONVENTION:
        # - Write-offs (WO_DebtSold, WO_Other): POSITIVE (absolute amounts)
        # - Provision increase: NEGATIVE (charge to P&L)
        # - Provision decrease: POSITIVE (release/benefit to P&L)
        # - DS_Provision_Release: POSITIVE (income/benefit)
        # - DS_Proceeds: POSITIVE (income/benefit)
        # - Gross Impairment: NEGATIVE = charge, POSITIVE = benefit
        #
        # Core coverage is back-solved in post-processing for months BEFORE debt sales
        # =======================================================================

        # Scaled WO_DebtSold is used throughout GBV mechanics (provision release, proceeds)
        debt_sale_wo_raw = wo_debt_sold  # wo_debt_sold is already scaled; used in provision calc
        ds_coverage_ratio = Config.DS_COVERAGE_RATIO  # Fixed 78.5%
        ds_proceeds_rate = Config.DS_PROCEEDS_RATE  # Fixed 24p per £1 of GBV sold

        # Step 1: Calculate total provision balance (Closing GBV × Coverage Ratio)
        total_coverage_ratio = imp_rates.get('Total_Coverage_Ratio', 0.12)
        total_provision_balance = closing_gbv * total_coverage_ratio

        # Step 2: Calculate provision movement
        total_provision_movement = total_provision_balance - prior_provision

        # Step 3: Calculate DS provision release (DS Coverage Ratio × DS WriteOffs)
        # Stored as POSITIVE (benefit - release of provision)
        ds_provision_release = ds_coverage_ratio * debt_sale_wo_raw

        # Step 4: Calculate DS proceeds (DS Proceeds Rate × DS WriteOffs)
        # Stored as POSITIVE (benefit - cash received)
        ds_proceeds = ds_proceeds_rate * debt_sale_wo_raw

        # Store write-offs as POSITIVE (absolute amounts)
        # wo_debt_sold is the scaled value; wo_debt_sold_raw is the pre-scale value
        wo_debt_sold_stored = wo_debt_sold   # scaled (used in GBV bridge & impairment)
        wo_other_stored = wo_other

        # Step 5: Calculate Non-DS provision movement
        # Non_DS = Total + DS_Release (add back the release to isolate non-DS movement)
        non_ds_provision_movement = total_provision_movement + ds_provision_release

        # Step 6: Calculate Gross impairment (excluding debt sales)
        # = NEGATED provision movement - WO_Other
        # P&L convention: provision increase = charge (negative), provision decrease = release (positive)
        # WO_Other is stored as positive, so subtract to represent expense
        gross_impairment_excl_ds = -non_ds_provision_movement - wo_other_stored

        # Step 7: Calculate Debt sale impact (gain/loss from debt sale)
        # = -WriteOffs (expense) + Release (positive) + Proceeds (positive)
        debt_sale_impact = -wo_debt_sold_stored + ds_provision_release + ds_proceeds

        # Step 8: Calculate Net impairment
        net_impairment = gross_impairment_excl_ds + debt_sale_impact

        # Calculate closing NBV
        closing_nbv = closing_gbv - total_provision_balance  # NBV = GBV - Provision

        # Build output row
        output_row = {
            'ForecastMonth': forecast_month,
            'Segment': segment,
            'Cohort': cohort,
            'MOB': mob,
            'OpeningGBV': round(opening_gbv, 2),

            # Rates
            'Coll_Principal_Rate': rates.get('Coll_Principal_Rate', 0.0),
            'Coll_Principal_Approach': rates.get('Coll_Principal_Approach', ''),
            'Coll_Interest_Rate': rates.get('Coll_Interest_Rate', 0.0),
            'Coll_Interest_Approach': rates.get('Coll_Interest_Approach', ''),
            'InterestRevenue_Rate': rates.get('InterestRevenue_Rate', 0.0),
            'InterestRevenue_Approach': rates.get('InterestRevenue_Approach', ''),
            'WO_DebtSold_Rate': rates.get('WO_DebtSold_Rate', 0.0),
            'WO_DebtSold_Approach': rates.get('WO_DebtSold_Approach', ''),
            'WO_DebtSold_ScaleFactor': round(ds_scale_factor, 6),
            'WO_Other_Rate': rates.get('WO_Other_Rate', 0.0),
            'WO_Other_Approach': rates.get('WO_Other_Approach', ''),
            'NewLoanAmount_Rate': rates.get('NewLoanAmount_Rate', 0.0),
            'NewLoanAmount_Approach': rates.get('NewLoanAmount_Approach', ''),
            'ContraSettlements_Principal_Rate': rates.get('ContraSettlements_Principal_Rate', 0.0),
            'ContraSettlements_Principal_Approach': rates.get('ContraSettlements_Principal_Approach', ''),
            'ContraSettlements_Interest_Rate': rates.get('ContraSettlements_Interest_Rate', 0.0),
            'ContraSettlements_Interest_Approach': rates.get('ContraSettlements_Interest_Approach', ''),

            # Amounts
            'NewLoanAmount': round(new_loan_amount, 2),
            'Coll_Principal': round(coll_principal, 2),
            'Coll_Interest': round(coll_interest, 2),
            'InterestRevenue': round(interest_revenue, 2),
            'WO_DebtSold_Raw': round(wo_debt_sold_raw, 2),    # pre-scale (donor curve × GBV)
            'WO_DebtSold': round(wo_debt_sold_stored, 2),  # POSITIVE scaled amount (used in GBV bridge)
            'WO_Other': round(wo_other_stored, 2),  # POSITIVE (absolute amount)
            'ContraSettlements_Principal': round(contra_principal, 2),
            'ContraSettlements_Interest': round(contra_interest, 2),

            # GBV
            'ClosingGBV': round(closing_gbv, 2),

            # Impairment - with full transparency breakdown
            'Total_Coverage_Ratio_Base': round(imp_rates.get('Total_Coverage_Ratio_Base', total_coverage_ratio), 6),
            'Seasonal_Factor': round(imp_rates.get('Seasonal_Factor', 1.0), 4),
            'Total_Coverage_Ratio': round(total_coverage_ratio, 6),
            'Total_Coverage_Approach': imp_rates.get('Total_Coverage_Approach', ''),

            # ShapeBorrowScaled traceability (only populated when ShapeBorrowScaled approach is used)
            'ShapeBorrow_Donor': imp_rates.get('ShapeBorrow_Donor', ''),
            'ShapeBorrow_Mode': imp_rates.get('ShapeBorrow_Mode', ''),
            'ShapeBorrow_RefWindow': imp_rates.get('ShapeBorrow_RefWindow', ''),
            'ShapeBorrow_Lc': round(imp_rates.get('ShapeBorrow_Lc', 0), 6) if imp_rates.get('ShapeBorrow_Lc') is not None else '',
            'ShapeBorrow_DonorMeanRef': round(imp_rates.get('ShapeBorrow_DonorMeanRef', 0), 6) if imp_rates.get('ShapeBorrow_DonorMeanRef') is not None else '',
            'ShapeBorrow_DonorRateAtMOB': round(imp_rates.get('ShapeBorrow_DonorRateAtMOB', 0), 6) if imp_rates.get('ShapeBorrow_DonorRateAtMOB') is not None else '',
            'ShapeBorrow_ShapeAdj': round(imp_rates.get('ShapeBorrow_ShapeAdj', 0), 6) if imp_rates.get('ShapeBorrow_ShapeAdj') is not None else '',
            'ShapeBorrow_FinalRate': round(imp_rates.get('ShapeBorrow_FinalRate', 0), 6) if imp_rates.get('ShapeBorrow_FinalRate') is not None else '',
            'ShapeBorrow_Error': imp_rates.get('ShapeBorrow_Error', ''),

            'Total_Provision_Balance': round(total_provision_balance, 2),
            'Prior_Provision_Balance': round(prior_provision, 2),
            'Total_Provision_Movement': round(total_provision_movement, 2),

            # Debt Sale - only occurs in debt sale months (Mar, Jun, Sep, Dec)
            'Is_Debt_Sale_Month': is_debt_sale_month(forecast_month),
            'Debt_Sale_WriteOffs': round(wo_debt_sold_stored, 2),  # POSITIVE (absolute amount)
            'Debt_Sale_Coverage_Ratio': round(ds_coverage_ratio, 6),
            'Debt_Sale_Proceeds_Rate': round(ds_proceeds_rate, 6),
            'Debt_Sale_Provision_Release': round(ds_provision_release, 2),
            'Debt_Sale_Proceeds': round(ds_proceeds, 2),

            # Net impairment components
            'Non_DS_Provision_Movement': round(non_ds_provision_movement, 2),
            'Gross_Impairment_ExcludingDS': round(gross_impairment_excl_ds, 2),
            'Debt_Sale_Impact': round(debt_sale_impact, 2),
            'Net_Impairment': round(net_impairment, 2),

            # NBV = GBV - Provision
            'ClosingNBV': round(closing_nbv, 2),
        }

        outputs.append(output_row)

        # Prepare next seed (if closing GBV > 0)
        if closing_gbv > 0:
            next_forecast_month = end_of_month(forecast_month + relativedelta(months=1))
            next_seeds.append({
                'Segment': segment,
                'Cohort': cohort,
                'MOB': mob + 1,
                'BoM': closing_gbv,
                'ForecastMonth': next_forecast_month,
                'Prior_Provision_Balance': total_provision_balance,
            })

    step_output = pd.DataFrame(outputs)
    next_seed = pd.DataFrame(next_seeds)

    return step_output, next_seed


def calculate_core_coverage_pre_debt_sale(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Back-solve core coverage ratio for months immediately BEFORE debt sale months.

    Per user specification Section 2D:
    For the month immediately before a debt sale month:
    - Implied_DS_Provision = (next month's) DS_Coverage_Ratio × (next month's) DS_WriteOffs
    - Core_Coverage = (Total_Provision - Implied_DS_Provision) / (Total_GBV - next month's DS_WriteOffs)

    This represents the implied coverage on the "core" portfolio (loans you're keeping)
    versus the "debt sale pool" (loans you'll sell next month at DS_Coverage_Ratio).

    Args:
        forecast: Complete forecast DataFrame with all months

    Returns:
        pd.DataFrame: Forecast with Core_Coverage columns added for pre-DS months
    """
    if len(forecast) == 0:
        return forecast

    df = forecast.copy()

    # Initialize core coverage columns
    df['Is_Pre_Debt_Sale_Month'] = False
    df['Next_Month_DS_WriteOffs'] = 0.0
    df['Implied_DS_Provision_In_Balance'] = 0.0
    df['Core_GBV'] = 0.0
    df['Core_Provision'] = 0.0
    df['Core_Coverage_Ratio'] = 0.0

    # Get unique forecast months sorted
    forecast_months = sorted(df['ForecastMonth'].unique())

    # For each segment × cohort combination
    for (segment, cohort), group in df.groupby(['Segment', 'Cohort']):
        group_sorted = group.sort_values('ForecastMonth')
        indices = group_sorted.index.tolist()

        for i, idx in enumerate(indices):
            current_month = df.loc[idx, 'ForecastMonth']

            # Check if NEXT month is a debt sale month
            if i + 1 < len(indices):
                next_idx = indices[i + 1]
                next_month = df.loc[next_idx, 'ForecastMonth']
                next_month_is_ds = df.loc[next_idx, 'Is_Debt_Sale_Month']
                next_month_ds_writeoffs = df.loc[next_idx, 'Debt_Sale_WriteOffs']

                if next_month_is_ds and next_month_ds_writeoffs > 0:
                    # This is a month BEFORE a debt sale - calculate core coverage
                    df.loc[idx, 'Is_Pre_Debt_Sale_Month'] = True
                    df.loc[idx, 'Next_Month_DS_WriteOffs'] = next_month_ds_writeoffs

                    # Get current month values
                    total_provision = df.loc[idx, 'Total_Provision_Balance']
                    total_gbv = df.loc[idx, 'ClosingGBV']
                    ds_coverage_ratio = Config.DS_COVERAGE_RATIO

                    # Calculate implied DS provision sitting in the balance
                    implied_ds_provision = ds_coverage_ratio * next_month_ds_writeoffs
                    df.loc[idx, 'Implied_DS_Provision_In_Balance'] = round(implied_ds_provision, 2)

                    # Calculate core values (back-solved)
                    core_gbv = total_gbv - next_month_ds_writeoffs
                    core_provision = total_provision - implied_ds_provision

                    df.loc[idx, 'Core_GBV'] = round(core_gbv, 2)
                    df.loc[idx, 'Core_Provision'] = round(core_provision, 2)

                    # Back-solve core coverage ratio
                    if core_gbv > 0:
                        core_coverage = core_provision / core_gbv
                        df.loc[idx, 'Core_Coverage_Ratio'] = round(core_coverage, 6)

    # Log summary
    pre_ds_count = df['Is_Pre_Debt_Sale_Month'].sum()
    if pre_ds_count > 0:
        logger.info(f"Calculated core coverage for {pre_ds_count} pre-debt-sale month rows")

    return df


def run_forecast(seed: pd.DataFrame, rate_lookup: pd.DataFrame,
                 impairment_lookup: pd.DataFrame, max_months: int) -> pd.DataFrame:
    """
    Run complete forecast loop.

    Args:
        seed: Starting seed
        rate_lookup: Rate lookup table
        impairment_lookup: Impairment lookup table
        max_months: Forecast horizon

    Returns:
        pd.DataFrame: Complete forecast
    """
    logger.info(f"Running forecast for {max_months} months...")

    all_outputs = []
    current_seed = seed.copy()

    for month in range(max_months):
        if len(current_seed) == 0:
            logger.info(f"No more active cohorts at month {month + 1}")
            break

        logger.info(f"Forecasting month {month + 1} with {len(current_seed)} cohorts")

        step_output, next_seed = run_one_step(current_seed, rate_lookup, impairment_lookup)

        if len(step_output) > 0:
            all_outputs.append(step_output)

        current_seed = next_seed

    if not all_outputs:
        logger.warning("No forecast output generated")
        return pd.DataFrame()

    forecast = pd.concat(all_outputs, ignore_index=True)
    forecast = forecast.sort_values(['ForecastMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    # Calculate core coverage for months immediately before debt sales (back-solve)
    forecast = calculate_core_coverage_pre_debt_sale(forecast)

    # Apply metric overlays if enabled (adjustments to final output amounts)
    if Config.ENABLE_OVERLAYS:
        forecast = apply_metric_overlays(forecast)

    logger.info(f"Forecast complete with {len(forecast)} rows")
    return forecast


# =============================================================================
# SECTION 12: OUTPUT GENERATION FUNCTIONS
# =============================================================================

def generate_summary_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level summary for Excel.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Summary by ForecastMonth and Segment
    """
    logger.info("Generating summary output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    agg_dict = {
        'OpeningGBV': 'sum',
        'InterestRevenue': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'ClosingGBV': 'sum',
        'Total_Provision_Balance': 'sum',
        'Net_Impairment': 'sum',
        'ClosingNBV': 'sum',
    }

    summary = forecast.groupby(['ForecastMonth', 'Segment']).agg(agg_dict).reset_index()

    # Calculate weighted coverage ratio
    summary['Total_Coverage_Ratio'] = summary.apply(
        lambda r: safe_divide(r['Total_Provision_Balance'], r['ClosingGBV']), axis=1
    )

    # Select and order columns
    columns = [
        'ForecastMonth', 'Segment', 'OpeningGBV', 'InterestRevenue',
        'Coll_Principal', 'Coll_Interest', 'WO_DebtSold', 'WO_Other',
        'ClosingGBV', 'Total_Coverage_Ratio', 'Net_Impairment', 'ClosingNBV'
    ]

    summary = summary[columns].sort_values(['ForecastMonth', 'Segment']).reset_index(drop=True)

    # Round numeric columns
    for col in summary.columns:
        if col not in ['ForecastMonth', 'Segment']:
            summary[col] = summary[col].round(2)

    logger.info(f"Generated summary with {len(summary)} rows")
    return summary


def generate_details_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete forecast for Excel.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Detailed forecast
    """
    logger.info("Generating details output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    details = forecast.copy()

    # Format dates
    details['ForecastMonth'] = pd.to_datetime(details['ForecastMonth']).dt.strftime('%Y-%m-%d')

    details = details.sort_values(['ForecastMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    logger.info(f"Generated details with {len(details)} rows")
    return details


def generate_impairment_output(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Create impairment-specific analysis.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        pd.DataFrame: Impairment analysis
    """
    logger.info("Generating impairment output...")

    if len(forecast) == 0:
        return pd.DataFrame()

    columns = [
        'ForecastMonth', 'Segment', 'Cohort', 'MOB', 'OpeningGBV', 'ClosingGBV',
        'Total_Coverage_Ratio', 'Total_Provision_Balance', 'Prior_Provision_Balance',
        'Total_Provision_Movement',
        # Debt Sale metrics
        'Is_Debt_Sale_Month', 'WO_DebtSold', 'Debt_Sale_WriteOffs', 'Debt_Sale_Coverage_Ratio',
        'Debt_Sale_Provision_Release', 'Debt_Sale_Proceeds',
        # Net impairment components
        'Non_DS_Provision_Movement', 'Gross_Impairment_ExcludingDS',
        'Debt_Sale_Impact', 'Net_Impairment',
        # NBV
        'ClosingNBV',
        # Core values (back-solved for pre-DS months only)
        'Is_Pre_Debt_Sale_Month', 'Next_Month_DS_WriteOffs', 'Implied_DS_Provision_In_Balance',
        'Core_GBV', 'Core_Provision', 'Core_Coverage_Ratio'
    ]

    impairment = forecast[columns].copy()
    impairment['ForecastMonth'] = pd.to_datetime(impairment['ForecastMonth']).dt.strftime('%Y-%m-%d')
    impairment = impairment.sort_values(['ForecastMonth', 'Segment', 'Cohort']).reset_index(drop=True)

    logger.info(f"Generated impairment output with {len(impairment)} rows")
    return impairment


def generate_validation_output(forecast: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create validation checks.

    Args:
        forecast: Complete forecast DataFrame

    Returns:
        tuple: (reconciliation_df, validation_checks_df)
    """
    logger.info("Generating validation output...")

    if len(forecast) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Reconciliation check
    recon = forecast.copy()

    # GBV reconciliation
    # WO_DebtSold and WO_Other are stored as POSITIVE (absolute amounts)
    recon['ClosingGBV_Calculated'] = (
        recon['OpeningGBV'] +
        recon['InterestRevenue'] -
        abs(recon['Coll_Principal']) -
        abs(recon['Coll_Interest']) -
        recon['WO_DebtSold'] -
        recon['WO_Other']
    ).round(2)

    recon['GBV_Variance'] = (recon['ClosingGBV_Calculated'] - recon['ClosingGBV']).abs().round(2)
    # Use tolerance of 1.0 to account for floating point rounding on large numbers
    recon['GBV_Status'] = recon['GBV_Variance'].apply(lambda x: 'PASS' if x < 1.0 else 'FAIL')

    # NBV reconciliation (NBV = GBV - Provision Balance)
    recon['ClosingNBV_Calculated'] = (recon['ClosingGBV'] - recon['Total_Provision_Balance']).round(2)
    recon['NBV_Variance'] = (recon['ClosingNBV_Calculated'] - recon['ClosingNBV']).abs().round(2)
    recon['NBV_Status'] = recon['NBV_Variance'].apply(lambda x: 'PASS' if x < 1.0 else 'FAIL')

    # Select reconciliation columns
    recon_cols = [
        'ForecastMonth', 'Segment', 'Cohort', 'OpeningGBV', 'InterestRevenue',
        'Coll_Principal', 'Coll_Interest', 'WO_DebtSold', 'WO_Other',
        'ClosingGBV_Calculated', 'ClosingGBV', 'GBV_Variance', 'GBV_Status',
        'Net_Impairment', 'ClosingNBV_Calculated', 'ClosingNBV', 'NBV_Variance', 'NBV_Status'
    ]

    reconciliation = recon[recon_cols].copy()
    reconciliation['ForecastMonth'] = pd.to_datetime(reconciliation['ForecastMonth']).dt.strftime('%Y-%m-%d')

    # Validation checks summary
    total_rows = len(forecast)

    checks = [
        {
            'Check': 'GBV_Reconciliation',
            'Total_Rows': total_rows,
            'Passed': (recon['GBV_Status'] == 'PASS').sum(),
            'Failed': (recon['GBV_Status'] == 'FAIL').sum(),
        },
        {
            'Check': 'NBV_Reconciliation',
            'Total_Rows': total_rows,
            'Passed': (recon['NBV_Status'] == 'PASS').sum(),
            'Failed': (recon['NBV_Status'] == 'FAIL').sum(),
        },
        {
            'Check': 'No_NaN_Values',
            'Total_Rows': total_rows,
            'Passed': total_rows - forecast[['OpeningGBV', 'ClosingGBV', 'ClosingNBV']].isna().any(axis=1).sum(),
            'Failed': forecast[['OpeningGBV', 'ClosingGBV', 'ClosingNBV']].isna().any(axis=1).sum(),
        },
        {
            'Check': 'No_Infinite_Values',
            'Total_Rows': total_rows,
            'Passed': total_rows - np.isinf(forecast.select_dtypes(include=[np.number])).any(axis=1).sum(),
            'Failed': np.isinf(forecast.select_dtypes(include=[np.number])).any(axis=1).sum(),
        },
        {
            'Check': 'Coverage_Ratio_Range',
            'Total_Rows': total_rows,
            # Allow coverage ratios between configured min and max (default 0-250%)
            # Higher cap accommodates IFRS 9 uplifts and conservative provisioning
            'Passed': ((forecast['Total_Coverage_Ratio'] >= Config.RATE_CAPS['Total_Coverage_Ratio'][0]) &
                      (forecast['Total_Coverage_Ratio'] <= Config.RATE_CAPS['Total_Coverage_Ratio'][1])).sum(),
            'Failed': ((forecast['Total_Coverage_Ratio'] < Config.RATE_CAPS['Total_Coverage_Ratio'][0]) |
                      (forecast['Total_Coverage_Ratio'] > Config.RATE_CAPS['Total_Coverage_Ratio'][1])).sum(),
        },
    ]

    validation_df = pd.DataFrame(checks)
    validation_df['Pass_Rate'] = (validation_df['Passed'] / validation_df['Total_Rows'] * 100).round(1).astype(str) + '%'
    validation_df['Status'] = validation_df.apply(
        lambda r: 'PASS' if r['Failed'] == 0 else 'FAIL', axis=1
    )

    # Overall status
    overall_passed = validation_df['Passed'].sum()
    overall_total = validation_df['Total_Rows'].sum()
    overall_failed = validation_df['Failed'].sum()
    overall_status = 'PASS' if overall_failed == 0 else 'FAIL'

    validation_df = pd.concat([
        validation_df,
        pd.DataFrame([{
            'Check': 'Overall',
            'Total_Rows': overall_total,
            'Passed': overall_passed,
            'Failed': overall_failed,
            'Pass_Rate': f"{overall_passed / overall_total * 100:.1f}%" if overall_total > 0 else '0%',
            'Status': overall_status,
        }])
    ], ignore_index=True)

    logger.info(f"Generated validation output - Overall status: {overall_status}")
    return reconciliation, validation_df


def generate_combined_actuals_forecast(fact_raw: pd.DataFrame, forecast: pd.DataFrame,
                                        output_dir: str) -> pd.DataFrame:
    """
    Generate a combined actuals + forecast output file for variance analysis.

    This creates a single file per iteration with both historical actuals and
    forecast data, enabling pivot table analysis and comparison to budget.

    Args:
        fact_raw: Historical actuals data from Fact_Raw
        forecast: Forecast data from the model
        output_dir: Output directory path

    Returns:
        pd.DataFrame: Combined actuals + forecast data
    """
    logger.info("Generating combined actuals + forecast output...")

    # Define common columns for both actuals and forecast
    common_cols = [
        'CalendarMonth', 'Segment', 'Cohort', 'MOB',
        'OpeningGBV', 'ClosingGBV', 'InterestRevenue',
        'Coll_Principal', 'Coll_Interest',
        'WO_DebtSold', 'WO_Other',
        'ContraSettlements_Principal', 'ContraSettlements_Interest',
        'NewLoanAmount'
    ]

    # Impairment columns (may not exist in older data)
    impairment_cols = [
        'Total_Provision_Balance', 'Total_Coverage_Ratio',
        'Total_Provision_Movement', 'Gross_Impairment_ExcludingDS',
        'Debt_Sale_Impact', 'Net_Impairment', 'ClosingNBV'
    ]

    # Process actuals
    actuals = fact_raw.copy()
    actuals['Source'] = 'Actuals'
    actuals['ForecastMonth'] = actuals['CalendarMonth']

    # Map ClosingGBV_Reported to ClosingGBV if needed
    if 'ClosingGBV_Reported' in actuals.columns and 'ClosingGBV' not in actuals.columns:
        actuals['ClosingGBV'] = actuals['ClosingGBV_Reported']

    # Map Provision_Balance to Total_Provision_Balance if available
    if 'Provision_Balance' in actuals.columns:
        actuals['Total_Provision_Balance'] = actuals['Provision_Balance']
        # Calculate coverage ratio for actuals
        actuals['Total_Coverage_Ratio'] = np.where(
            actuals['ClosingGBV'] > 0,
            actuals['Total_Provision_Balance'] / actuals['ClosingGBV'],
            0
        )
        # Calculate NBV
        actuals['ClosingNBV'] = actuals['ClosingGBV'] - actuals['Total_Provision_Balance']

    # Process forecast
    fcst = forecast.copy()
    fcst['Source'] = 'Forecast'
    fcst['CalendarMonth'] = fcst['ForecastMonth']

    # Get columns that exist in both
    actuals_cols = set(actuals.columns)
    forecast_cols = set(fcst.columns)

    # Build final column list
    final_cols = ['Source', 'CalendarMonth', 'Segment', 'Cohort', 'MOB']

    # Add financial columns that exist
    for col in ['OpeningGBV', 'ClosingGBV', 'InterestRevenue', 'Coll_Principal',
                'Coll_Interest', 'WO_DebtSold', 'WO_Other',
                'ContraSettlements_Principal', 'ContraSettlements_Interest',
                'NewLoanAmount', 'Total_Provision_Balance', 'Total_Coverage_Ratio',
                'Total_Provision_Movement', 'Gross_Impairment_ExcludingDS',
                'Debt_Sale_Impact', 'Net_Impairment', 'ClosingNBV']:
        # Add column if it exists in either dataset
        if col in actuals_cols or col in forecast_cols:
            final_cols.append(col)
            # Fill missing column with 0
            if col not in actuals.columns:
                actuals[col] = 0.0
            if col not in fcst.columns:
                fcst[col] = 0.0

    # Select and combine
    actuals_out = actuals[final_cols].copy()
    forecast_out = fcst[final_cols].copy()

    combined = pd.concat([actuals_out, forecast_out], ignore_index=True)

    # Sort by date, segment, cohort
    combined = combined.sort_values(['CalendarMonth', 'Segment', 'Cohort', 'MOB']).reset_index(drop=True)

    # Format date as string for Excel
    combined['CalendarMonth'] = pd.to_datetime(combined['CalendarMonth']).dt.strftime('%Y-%m-%d')

    # Round numeric columns
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        combined[col] = combined[col].round(2)

    logger.info(f"Generated combined output with {len(combined)} rows "
                f"({len(actuals_out)} actuals + {len(forecast_out)} forecast)")

    # Export to Excel
    output_path = os.path.join(output_dir, 'Combined_Actuals_Forecast.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        combined.to_excel(writer, sheet_name='Combined', index=False)

        # Add a summary sheet by month
        monthly_summary = combined.groupby(['Source', 'CalendarMonth']).agg({
            'OpeningGBV': 'sum',
            'ClosingGBV': 'sum',
            'InterestRevenue': 'sum',
            'Coll_Principal': 'sum',
            'Coll_Interest': 'sum',
            'WO_DebtSold': 'sum',
            'WO_Other': 'sum',
            'Total_Provision_Balance': 'sum' if 'Total_Provision_Balance' in combined.columns else 'first',
            'Net_Impairment': 'sum' if 'Net_Impairment' in combined.columns else 'first',
            'ClosingNBV': 'sum' if 'ClosingNBV' in combined.columns else 'first',
        }).reset_index()
        monthly_summary.to_excel(writer, sheet_name='Monthly_Summary', index=False)

        # Add a segment summary sheet
        segment_summary = combined.groupby(['Source', 'CalendarMonth', 'Segment']).agg({
            'OpeningGBV': 'sum',
            'ClosingGBV': 'sum',
            'InterestRevenue': 'sum',
            'Coll_Principal': 'sum',
            'Coll_Interest': 'sum',
            'WO_DebtSold': 'sum',
            'WO_Other': 'sum',
        }).reset_index()
        segment_summary.to_excel(writer, sheet_name='Segment_Summary', index=False)

    logger.info(f"Created: {output_path}")

    return combined


def build_ex_contra_actuals(fact_raw_full: pd.DataFrame, cutoff_date: str,
                            forecast_cohorts: pd.DataFrame) -> pd.DataFrame:
    """
    Build an ex-contra actuals series with month-by-month GBV roll-forward
    that mirrors the forecast bridge (excluding contra settlements).

    For each (Segment, Cohort), creates a time-ordered monthly series where:
    - OpeningGBV_ExContra[t0] = OpeningGBV_Reported[t0]
    - ClosingGBV_ExContra[t] = OpeningGBV_ExContra[t] + InterestRevenue
      + Coll_Principal + Coll_Interest - WO_DebtSold - WO_Other
    - OpeningGBV_ExContra[t+1] = ClosingGBV_ExContra[t]

    Then derives ex-contra provision and NBV using the actual coverage ratio.

    Args:
        fact_raw_full: Full historical data (unfiltered)
        cutoff_date: Cutoff date string in YYYY-MM format
        forecast_cohorts: DataFrame with Segment x Cohort combinations to include

    Returns:
        pd.DataFrame: Ex-contra actuals at Segment x Cohort x Month level
    """
    logger.info("Building ex-contra actuals series...")

    cutoff_dt = end_of_month(pd.Timestamp(cutoff_date + '-01'))
    cutoff_yyyymm = int(cutoff_date.replace('-', ''))

    # Filter to post-cutoff, BB cohorts, matching forecast cohorts
    actuals = fact_raw_full[fact_raw_full['CalendarMonth'] >= cutoff_dt].copy()
    actuals = actuals[actuals['Cohort'].astype(int) < cutoff_yyyymm].copy()
    actuals = actuals.merge(forecast_cohorts, on=['Segment', 'Cohort'], how='inner')

    if len(actuals) == 0:
        logger.warning("  No post-cutoff actuals for ex-contra series")
        return pd.DataFrame()

    # Map column names
    if 'ClosingGBV_Reported' in actuals.columns and 'ClosingGBV' not in actuals.columns:
        actuals['ClosingGBV'] = actuals['ClosingGBV_Reported']
    if 'Provision_Balance' in actuals.columns:
        actuals['Total_Provision_Balance'] = actuals['Provision_Balance'].abs()

    # Aggregate to Segment x Cohort x CalendarMonth level
    agg_cols = {
        'OpeningGBV': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'InterestRevenue': 'sum',
        'ClosingGBV': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'Total_Provision_Balance': 'sum',
    }
    if 'ContraSettlements_Principal' in actuals.columns:
        agg_cols['ContraSettlements_Principal'] = 'sum'
    if 'ContraSettlements_Interest' in actuals.columns:
        agg_cols['ContraSettlements_Interest'] = 'sum'

    actuals_agg = actuals.groupby(['Segment', 'Cohort', 'CalendarMonth']).agg(agg_cols).reset_index()
    actuals_agg = actuals_agg.sort_values(['Segment', 'Cohort', 'CalendarMonth']).reset_index(drop=True)

    # Build ex-contra roll-forward per cohort
    rows = []
    for (segment, cohort), group in actuals_agg.groupby(['Segment', 'Cohort']):
        group = group.sort_values('CalendarMonth').reset_index(drop=True)
        opening_ex_contra = None

        for i, row in group.iterrows():
            if opening_ex_contra is None:
                # First month: use reported opening GBV
                opening_ex_contra = row['OpeningGBV']

            # Ex-contra closing: same bridge as forecast (no contra settlements)
            closing_ex_contra = (
                opening_ex_contra +
                row['InterestRevenue'] +
                row['Coll_Principal'] +    # negative (inflow), so adding = subtracting
                row['Coll_Interest'] -     # negative (inflow), so adding = subtracting
                row['WO_DebtSold'] -
                row['WO_Other']
            )
            closing_ex_contra = max(0.0, closing_ex_contra)

            # Actual coverage ratio from reported values
            cr_actual = (row['Total_Provision_Balance'] / row['ClosingGBV']
                         if row['ClosingGBV'] > 0 else 0)

            # Ex-contra provision = actual CR x ex-contra GBV
            provision_ex_contra = cr_actual * closing_ex_contra

            # Ex-contra NBV
            nbv_ex_contra = closing_ex_contra - provision_ex_contra

            # Contra effect (sanity check): reported - ex-contra
            contra_effect = row['ClosingGBV'] - closing_ex_contra

            contra_principal = row.get('ContraSettlements_Principal', 0)
            contra_interest = row.get('ContraSettlements_Interest', 0)

            rows.append({
                'Segment': segment,
                'Cohort': cohort,
                'Month': row['CalendarMonth'],
                # Reported actuals
                'OpeningGBV_Reported': row['OpeningGBV'],
                'ClosingGBV_Reported': row['ClosingGBV'],
                'Total_Provision_Balance_Reported': row['Total_Provision_Balance'],
                # Movements (from actuals)
                'Coll_Principal': row['Coll_Principal'],
                'Coll_Interest': row['Coll_Interest'],
                'InterestRevenue': row['InterestRevenue'],
                'WO_DebtSold': row['WO_DebtSold'],
                'WO_Other': row['WO_Other'],
                'ContraSettlements_Principal': contra_principal,
                'ContraSettlements_Interest': contra_interest,
                # Ex-contra series
                'OpeningGBV_ExContra': round(opening_ex_contra, 2),
                'ClosingGBV_ExContra': round(closing_ex_contra, 2),
                'Total_Coverage_Ratio_Actual': round(cr_actual, 6),
                'Total_Provision_Balance_ExContra': round(provision_ex_contra, 2),
                'ClosingNBV_ExContra': round(nbv_ex_contra, 2),
                # Sanity check
                'Contra_Effect_on_GBV': round(contra_effect, 2),
            })

            # Next month's opening = this month's ex-contra closing
            opening_ex_contra = closing_ex_contra

    result = pd.DataFrame(rows)

    n_cohorts = result[['Segment', 'Cohort']].drop_duplicates().shape[0]
    n_months = result['Month'].nunique()
    logger.info(f"  Built ex-contra actuals: {n_cohorts} cohorts x {n_months} months = {len(result)} rows")

    return result


def generate_backtest_comparison(fact_raw_full: pd.DataFrame, forecast: pd.DataFrame,
                                  cutoff_date: str, use_ex_contra: bool = False) -> pd.DataFrame:
    """
    Generate a Segment x Cohort x Month backtest comparison of forecast vs actuals.

    Compares forecast output against post-cutoff actuals for the exact cohorts
    present in the forecast (BB cohorts only).

    When use_ex_contra=True, stock metrics (OpeningGBV, ClosingGBV,
    Total_Provision_Balance) use the ex-contra actuals series for a like-for-like
    comparison with the forecast bridge (which excludes contra settlements).
    Flow metrics (collections, interest, write-offs) remain as reported.

    Args:
        fact_raw_full: Full historical data (unfiltered, includes post-cutoff actuals)
        forecast: Forecast output from the model
        cutoff_date: Cutoff date string in YYYY-MM format
        use_ex_contra: If True, use ex-contra actuals for stock metrics

    Returns:
        pd.DataFrame: Backtest comparison with Forecast, Actual, Variance per metric
    """
    mode_label = "ex-contra" if use_ex_contra else "reported"
    logger.info(f"Generating backtest comparison (actuals mode: {mode_label})...")

    cutoff_dt = end_of_month(pd.Timestamp(cutoff_date + '-01'))
    cutoff_yyyymm = int(cutoff_date.replace('-', ''))

    # Get the Segment x Cohort combinations in the forecast
    forecast_cohorts = forecast[['Segment', 'Cohort']].drop_duplicates()
    logger.info(f"  Forecast contains {len(forecast_cohorts)} Segment x Cohort combinations")

    # Filter actuals to: post-cutoff months, BB cohorts only (matching forecast cohorts)
    actuals = fact_raw_full[fact_raw_full['CalendarMonth'] >= cutoff_dt].copy()
    actuals = actuals[actuals['Cohort'].astype(int) < cutoff_yyyymm].copy()
    actuals = actuals.merge(forecast_cohorts, on=['Segment', 'Cohort'], how='inner')

    if len(actuals) == 0:
        logger.warning("  No post-cutoff actuals found for backtest comparison")
        return pd.DataFrame()

    # Map actuals column names for consistency with forecast column names
    if 'ClosingGBV_Reported' in actuals.columns and 'ClosingGBV' not in actuals.columns:
        actuals['ClosingGBV'] = actuals['ClosingGBV_Reported']
    if 'Provision_Balance' in actuals.columns:
        actuals['Total_Provision_Balance'] = actuals['Provision_Balance'].abs()

    # Aggregate actuals by Segment x Cohort x CalendarMonth
    actuals_agg = actuals.groupby(['Segment', 'Cohort', 'CalendarMonth']).agg({
        'OpeningGBV': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'InterestRevenue': 'sum',
        'ClosingGBV': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
        'Total_Provision_Balance': 'sum',
    }).reset_index()

    # Compute actual coverage ratio and ClosingGBV_exclcontra
    actuals_agg['Total_Coverage_Ratio'] = np.where(
        actuals_agg['ClosingGBV'] > 0,
        actuals_agg['Total_Provision_Balance'] / actuals_agg['ClosingGBV'],
        0
    )
    actuals_agg['ClosingGBV_exclcontra'] = (
        actuals_agg['OpeningGBV'] +
        actuals_agg['InterestRevenue'] +
        actuals_agg['Coll_Principal'] +
        actuals_agg['Coll_Interest'] -
        actuals_agg['WO_DebtSold'] -
        actuals_agg['WO_Other']
    )

    # If using ex-contra mode, build the roll-forward series and overlay stock metrics
    if use_ex_contra:
        ex_contra = build_ex_contra_actuals(fact_raw_full, cutoff_date, forecast_cohorts)
        if len(ex_contra) > 0:
            # Merge ex-contra stock metrics into actuals_agg
            ec_lookup = ex_contra[['Segment', 'Cohort', 'Month',
                                    'OpeningGBV_ExContra', 'ClosingGBV_ExContra',
                                    'Total_Provision_Balance_ExContra',
                                    'ClosingNBV_ExContra']].copy()
            ec_lookup.rename(columns={'Month': 'CalendarMonth'}, inplace=True)
            actuals_agg = actuals_agg.merge(ec_lookup, on=['Segment', 'Cohort', 'CalendarMonth'], how='left')

            # Override stock metrics with ex-contra values
            actuals_agg['OpeningGBV'] = actuals_agg['OpeningGBV_ExContra'].fillna(actuals_agg['OpeningGBV'])
            actuals_agg['ClosingGBV'] = actuals_agg['ClosingGBV_ExContra'].fillna(actuals_agg['ClosingGBV'])
            actuals_agg['Total_Provision_Balance'] = actuals_agg['Total_Provision_Balance_ExContra'].fillna(
                actuals_agg['Total_Provision_Balance'])
            # Recompute coverage ratio on ex-contra basis
            actuals_agg['Total_Coverage_Ratio'] = np.where(
                actuals_agg['ClosingGBV'] > 0,
                actuals_agg['Total_Provision_Balance'] / actuals_agg['ClosingGBV'],
                0
            )
            # Recompute ClosingGBV_exclcontra (same as ClosingGBV in ex-contra mode)
            actuals_agg['ClosingGBV_exclcontra'] = actuals_agg['ClosingGBV']

            # Clean up temp columns
            actuals_agg.drop(columns=['OpeningGBV_ExContra', 'ClosingGBV_ExContra',
                                       'Total_Provision_Balance_ExContra', 'ClosingNBV_ExContra'],
                              inplace=True, errors='ignore')

    actuals_agg.rename(columns={'CalendarMonth': 'Month'}, inplace=True)

    # Aggregate forecast by Segment x Cohort x ForecastMonth
    forecast_agg_cols = {
        'OpeningGBV': 'sum',
        'Coll_Principal': 'sum',
        'Coll_Interest': 'sum',
        'InterestRevenue': 'sum',
        'ClosingGBV': 'sum',
        'WO_DebtSold': 'sum',
        'WO_Other': 'sum',
    }
    if 'Total_Provision_Balance' in forecast.columns:
        forecast_agg_cols['Total_Provision_Balance'] = 'sum'

    fcst_agg = forecast.groupby(['Segment', 'Cohort', 'ForecastMonth']).agg(
        forecast_agg_cols
    ).reset_index()

    # Compute forecast coverage ratio and ClosingGBV_exclcontra from aggregated values
    if 'Total_Provision_Balance' in fcst_agg.columns:
        fcst_agg['Total_Coverage_Ratio'] = np.where(
            fcst_agg['ClosingGBV'] > 0,
            fcst_agg['Total_Provision_Balance'] / fcst_agg['ClosingGBV'],
            0
        )
    fcst_agg['ClosingGBV_exclcontra'] = (
        fcst_agg['OpeningGBV'] +
        fcst_agg['InterestRevenue'] -
        abs(fcst_agg['Coll_Principal']) -
        abs(fcst_agg['Coll_Interest']) -
        fcst_agg['WO_DebtSold'] -
        fcst_agg['WO_Other']
    )

    fcst_agg.rename(columns={'ForecastMonth': 'Month'}, inplace=True)

    # Merge forecast and actuals
    metrics = ['OpeningGBV', 'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
               'ClosingGBV', 'ClosingGBV_exclcontra', 'WO_DebtSold', 'WO_Other']
    if 'Total_Provision_Balance' in fcst_agg.columns:
        metrics.append('Total_Provision_Balance')
    if 'Total_Coverage_Ratio' in fcst_agg.columns:
        metrics.append('Total_Coverage_Ratio')

    # Suffix: _Forecast and _Actual
    merged = fcst_agg.merge(
        actuals_agg,
        on=['Segment', 'Cohort', 'Month'],
        how='inner',
        suffixes=('_Forecast', '_Actual')
    )

    if len(merged) == 0:
        logger.warning("  No matching months between forecast and actuals for backtest")
        return pd.DataFrame()

    # Build the output: one row per Segment x Cohort x Month x Metric
    rows = []
    for _, row in merged.iterrows():
        for metric in metrics:
            fcst_col = f'{metric}_Forecast' if f'{metric}_Forecast' in row.index else metric
            act_col = f'{metric}_Actual' if f'{metric}_Actual' in row.index else metric

            fcst_val = row.get(fcst_col, 0)
            act_val = row.get(act_col, 0)

            variance = fcst_val - act_val
            pct_var = (variance / abs(act_val) * 100) if act_val != 0 else 0

            rows.append({
                'Segment': row['Segment'],
                'Cohort': row['Cohort'],
                'Month': row['Month'],
                'Metric': metric,
                'Forecast': round(fcst_val, 2),
                'Actual': round(act_val, 2),
                'Variance': round(variance, 2),
                'Pct_Variance': round(pct_var, 2),
                'Actuals_Basis': mode_label,
            })

    result = pd.DataFrame(rows)
    result = result.sort_values(['Segment', 'Cohort', 'Month', 'Metric']).reset_index(drop=True)

    n_months = result['Month'].nunique()
    n_cohorts = result[['Segment', 'Cohort']].drop_duplicates().shape[0]
    logger.info(f"  Backtest comparison ({mode_label}): {n_cohorts} cohorts x {n_months} months x {len(metrics)} metrics = {len(result)} rows")

    return result


def export_to_excel(summary: pd.DataFrame, details: pd.DataFrame,
                    impairment: pd.DataFrame, reconciliation: pd.DataFrame,
                    validation: pd.DataFrame, output_dir: str) -> None:
    """
    Write all outputs to Excel workbooks.

    Args:
        summary: Summary DataFrame
        details: Details DataFrame
        impairment: Impairment DataFrame
        reconciliation: Reconciliation DataFrame
        validation: Validation checks DataFrame
        output_dir: Output directory path
    """
    logger.info(f"Exporting to Excel in: {output_dir}")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Export Forecast_Summary.xlsx
    summary_path = os.path.join(output_dir, 'Forecast_Summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
    logger.info(f"Created: {summary_path}")

    # Export Forecast_Details.xlsx
    details_path = os.path.join(output_dir, 'Forecast_Details.xlsx')
    with pd.ExcelWriter(details_path, engine='openpyxl') as writer:
        details.to_excel(writer, sheet_name='All_Cohorts', index=False)
    logger.info(f"Created: {details_path}")

    # Export Impairment_Analysis.xlsx
    impairment_path = os.path.join(output_dir, 'Impairment_Analysis.xlsx')
    with pd.ExcelWriter(impairment_path, engine='openpyxl') as writer:
        impairment.to_excel(writer, sheet_name='Impairment_Detail', index=False)

        # Coverage ratios sheet
        if len(impairment) > 0:
            coverage_cols = ['Segment', 'Cohort', 'MOB', 'Total_Coverage_Ratio',
                           'Debt_Sale_Coverage_Ratio', 'Debt_Sale_Proceeds_Rate']
            coverage_cols = [c for c in coverage_cols if c in impairment.columns]
            coverage = impairment[coverage_cols].drop_duplicates()
            coverage.to_excel(writer, sheet_name='Coverage_Ratios', index=False)
    logger.info(f"Created: {impairment_path}")

    # Export Validation_Report.xlsx
    validation_path = os.path.join(output_dir, 'Validation_Report.xlsx')
    with pd.ExcelWriter(validation_path, engine='openpyxl') as writer:
        reconciliation.to_excel(writer, sheet_name='Reconciliation', index=False)
        validation.to_excel(writer, sheet_name='Validation_Checks', index=False)
    logger.info(f"Created: {validation_path}")

    logger.info("Excel export complete")


def generate_comprehensive_transparency_report(
    fact_raw: pd.DataFrame,
    methodology: pd.DataFrame,
    curves_base: pd.DataFrame,
    curves_extended: pd.DataFrame,
    rate_lookup: pd.DataFrame,
    impairment_lookup: pd.DataFrame,
    forecast: pd.DataFrame,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    impairment_output: pd.DataFrame,
    reconciliation: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str,
    max_months: int,
    backtest: Optional[pd.DataFrame] = None,
    ex_contra_actuals: Optional[pd.DataFrame] = None,
    ds_donor_map: Optional[pd.DataFrame] = None,
    ds_scale_factors_diag: Optional[pd.DataFrame] = None,
) -> str:
    """
    Generate single comprehensive Excel report with full audit trail and all outputs.

    Combines the transparency report (showing methodology/rates/curves) with
    all forecast outputs (summary, details, impairment, validation) in one file.

    Args:
        fact_raw: Raw historical data
        methodology: Rate methodology rules
        curves_base: Historical rate curves
        curves_extended: Extended rate curves
        rate_lookup: Rate lookup table
        impairment_lookup: Impairment lookup table
        forecast: Full forecast output
        summary: Summary output
        details: Details output
        impairment_output: Impairment output
        reconciliation: Reconciliation output
        validation: Validation output
        output_dir: Output directory
        max_months: Forecast horizon
        backtest: Optional backtest comparison DataFrame
        ex_contra_actuals: Optional ex-contra actuals series DataFrame

    Returns:
        str: Path to generated report
    """
    logger.info("Generating comprehensive transparency report...")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Forecast_Transparency_Report.xlsx')

    # ==========================================================================
    # Prepare Actuals Data
    # ==========================================================================
    actuals_df = fact_raw.copy()
    actuals_df['Coll_Principal_Rate'] = actuals_df.apply(
        lambda r: safe_divide(r['Coll_Principal'], r['OpeningGBV']), axis=1)
    actuals_df['Coll_Interest_Rate'] = actuals_df.apply(
        lambda r: safe_divide(r['Coll_Interest'], r['OpeningGBV']), axis=1)
    actuals_df['InterestRevenue_Rate_Annual'] = actuals_df.apply(
        lambda r: safe_divide(r['InterestRevenue'], r['OpeningGBV']) * safe_divide(365, r['DaysInMonth'], 12), axis=1)
    actuals_df['WO_DebtSold_Rate'] = actuals_df.apply(
        lambda r: safe_divide(r['WO_DebtSold'], r['OpeningGBV']), axis=1)
    actuals_df['WO_Other_Rate'] = actuals_df.apply(
        lambda r: safe_divide(r['WO_Other'], r['OpeningGBV']), axis=1)
    actuals_df['Coverage_Ratio'] = actuals_df.apply(
        lambda r: safe_divide(r['Provision_Balance'], r['ClosingGBV_Reported']), axis=1)
    actuals_df['GBV_Runoff_Rate'] = actuals_df.apply(
        lambda r: safe_divide(r['OpeningGBV'] - r['ClosingGBV_Reported'], r['OpeningGBV']), axis=1)

    # ClosingGBV excluding contra settlements
    actuals_df['ClosingGBV_exclcontra'] = (
        actuals_df['OpeningGBV'] +
        actuals_df['InterestRevenue'] +
        actuals_df['Coll_Principal'] +
        actuals_df['Coll_Interest'] -
        actuals_df['WO_DebtSold'] -
        actuals_df['WO_Other']
    )

    actuals_cols = [
        'CalendarMonth', 'Segment', 'Cohort', 'MOB',
        'OpeningGBV', 'Coll_Principal', 'Coll_Interest', 'InterestRevenue',
        'WO_DebtSold', 'WO_Other',
        'ContraSettlements_Principal', 'ContraSettlements_Interest',
        'ClosingGBV_Reported', 'ClosingGBV_exclcontra', 'Provision_Balance',
        'Coll_Principal_Rate', 'Coll_Interest_Rate', 'InterestRevenue_Rate_Annual',
        'WO_DebtSold_Rate', 'WO_Other_Rate', 'Coverage_Ratio', 'GBV_Runoff_Rate'
    ]
    actuals_output = actuals_df[[c for c in actuals_cols if c in actuals_df.columns]].copy()
    actuals_output['DataType'] = 'Actual'

    # ==========================================================================
    # Prepare Historical Curves
    # ==========================================================================
    curves_cols = [
        'Segment', 'Cohort', 'MOB', 'OpeningGBV', 'ClosingGBV_Reported',
        'Coll_Principal_Rate', 'Coll_Interest_Rate', 'InterestRevenue_Rate',
        'WO_DebtSold_Rate', 'WO_Other_Rate', 'Total_Coverage_Ratio'
    ]
    for col in curves_cols:
        if col not in curves_base.columns:
            curves_base[col] = 0.0
    historical_curves = curves_base[[c for c in curves_cols if c in curves_base.columns]].copy()
    historical_curves['CurveType'] = 'Historical'

    # ==========================================================================
    # Prepare Extended Curves
    # ==========================================================================
    extended_curves = curves_extended[[c for c in curves_cols if c in curves_extended.columns]].copy()
    max_historical_mob = curves_base.groupby(['Segment', 'Cohort'])['MOB'].max().reset_index()
    max_historical_mob.columns = ['Segment', 'Cohort', 'Max_Historical_MOB']
    extended_curves = extended_curves.merge(max_historical_mob, on=['Segment', 'Cohort'], how='left')
    extended_curves['CurveType'] = extended_curves.apply(
        lambda r: 'Extended' if r['MOB'] > r.get('Max_Historical_MOB', 0) else 'Historical', axis=1)
    if 'Max_Historical_MOB' in extended_curves.columns:
        extended_curves = extended_curves.drop(columns=['Max_Historical_MOB'])

    # ==========================================================================
    # Prepare Combined View
    # ==========================================================================
    actuals_rows = []
    for _, row in actuals_df.iterrows():
        actuals_rows.append({
            'Month': row['CalendarMonth'],
            'Segment': row['Segment'],
            'Cohort': row['Cohort'],
            'MOB': row['MOB'],
            'DataType': 'Actual',
            'OpeningGBV': row['OpeningGBV'],
            'Coll_Principal': row['Coll_Principal'],
            'Coll_Interest': row['Coll_Interest'],
            'InterestRevenue': row['InterestRevenue'],
            'WO_DebtSold': row['WO_DebtSold'],
            'WO_Other': row['WO_Other'],
            'ClosingGBV': row['ClosingGBV_Reported'],
            'Coll_Principal_Rate': row.get('Coll_Principal_Rate', 0),
            'Coll_Interest_Rate': row.get('Coll_Interest_Rate', 0),
            'InterestRevenue_Rate': row.get('InterestRevenue_Rate_Annual', 0),
            'WO_DebtSold_Rate': row.get('WO_DebtSold_Rate', 0),
            'WO_Other_Rate': row.get('WO_Other_Rate', 0),
            'Provision_Balance': row.get('Provision_Balance', 0),
            'Total_Coverage_Ratio': row.get('Coverage_Ratio', 0),
        })

    forecast_rows = []
    for _, row in forecast.iterrows():
        forecast_rows.append({
            'Month': row['ForecastMonth'],
            'Segment': row['Segment'],
            'Cohort': row['Cohort'],
            'MOB': row['MOB'],
            'DataType': 'Forecast',
            'OpeningGBV': row['OpeningGBV'],
            'Coll_Principal': row['Coll_Principal'],
            'Coll_Interest': row['Coll_Interest'],
            'InterestRevenue': row['InterestRevenue'],
            'WO_DebtSold': row['WO_DebtSold'],
            'WO_Other': row['WO_Other'],
            'ClosingGBV': row['ClosingGBV'],
            'Coll_Principal_Rate': row.get('Coll_Principal_Rate', 0),
            'Coll_Interest_Rate': row.get('Coll_Interest_Rate', 0),
            'InterestRevenue_Rate': row.get('InterestRevenue_Rate', 0),
            'WO_DebtSold_Rate': row.get('WO_DebtSold_Rate', 0),
            'WO_Other_Rate': row.get('WO_Other_Rate', 0),
            'Provision_Balance': row.get('Total_Provision_Balance', 0),
            'Total_Coverage_Ratio': row.get('Total_Coverage_Ratio', 0),
        })

    combined_df = pd.DataFrame(actuals_rows + forecast_rows)
    combined_df = combined_df.sort_values(['Segment', 'Cohort', 'Month']).reset_index(drop=True)

    # ==========================================================================
    # Prepare Curve Comparison (actual historical rates vs forecast-applied rates)
    # ==========================================================================
    flow_metrics = ['Coll_Principal', 'Coll_Interest', 'InterestRevenue']
    curve_rows = []

    for metric in flow_metrics:
        rate_col = f'{metric}_Rate'
        approach_col = f'{metric}_Approach'

        if rate_col not in historical_curves.columns:
            continue

        actual_slice = historical_curves[['Segment', 'Cohort', 'MOB', rate_col]].copy()
        actual_slice.rename(columns={rate_col: 'Actual_Historical_Rate'}, inplace=True)

        # Derive a calendar month for each actual row (Cohort YYYYMM + MOB months)
        # so that actual-only rows have a Forecast_Month value for alignment.
        # Use integer year/month arithmetic to avoid deprecated 'M' timedelta unit.
        cohort_int = actual_slice['Cohort'].astype(int)
        cohort_year = (cohort_int // 100).astype(int)
        cohort_month = (cohort_int % 100).astype(int)
        mob_int = actual_slice['MOB'].astype(int)
        total_months = cohort_year * 12 + cohort_month - 1 + mob_int  # 0-based month index
        fm_year = total_months // 12
        fm_month = total_months % 12 + 1  # back to 1-based
        actual_slice['Forecast_Month'] = pd.to_datetime(
            {'year': fm_year, 'month': fm_month, 'day': 1}
        ) + pd.offsets.MonthEnd(0)
        actual_slice['Forecast_Approach'] = 'Actual'

        # Use forecast output (not rate_lookup) so the comparison carries
        # the actual forecast calendar month for each cohort/MOB point.
        if rate_col not in forecast.columns:
            continue

        forecast_cols = ['Segment', 'Cohort', 'MOB', 'ForecastMonth', rate_col]
        if approach_col in forecast.columns:
            forecast_cols.append(approach_col)

        forecast_slice = forecast[forecast_cols].copy()
        forecast_slice.rename(columns={rate_col: 'Forecast_Applied_Rate'}, inplace=True)
        forecast_slice.rename(columns={'ForecastMonth': 'Forecast_Month'}, inplace=True)
        if approach_col in forecast_slice.columns:
            forecast_slice.rename(columns={approach_col: 'Forecast_Approach'}, inplace=True)
        else:
            forecast_slice['Forecast_Approach'] = ''

        # Outer merge so actual-only rows (pre-cutoff MOBs) are included in full,
        # not just the MOBs that happen to also have a forecast row.
        # Rows with only actuals get Forecast_Applied_Rate = NaN (and vice versa).
        merged = pd.merge(
            forecast_slice,
            actual_slice[['Segment', 'Cohort', 'MOB', 'Actual_Historical_Rate',
                           'Forecast_Month', 'Forecast_Approach']],
            on=['Segment', 'Cohort', 'MOB'],
            how='outer',
            suffixes=('', '_actual')
        )
        # For actual-only rows, fill Forecast_Month and Forecast_Approach from the actual side
        merged['Forecast_Month'] = merged['Forecast_Month'].combine_first(
            merged['Forecast_Month_actual']
        )
        merged['Forecast_Approach'] = merged['Forecast_Approach'].combine_first(
            merged['Forecast_Approach_actual']
        )
        # Drop the helper columns brought in by the outer merge suffixes
        merged.drop(columns=[c for c in merged.columns if c.endswith('_actual')], inplace=True)

        merged['Metric'] = metric
        merged['Rate_Delta_Forecast_minus_Actual'] = (
            merged['Forecast_Applied_Rate'] - merged['Actual_Historical_Rate']
        )
        curve_rows.append(merged)

    if curve_rows:
        curve_comparison = pd.concat(curve_rows, ignore_index=True)
        curve_comparison = curve_comparison[
            ['Segment', 'Cohort', 'Metric', 'MOB',
             'Actual_Historical_Rate', 'Forecast_Applied_Rate',
             'Rate_Delta_Forecast_minus_Actual',
             'Forecast_Month', 'Forecast_Approach']
        ]
        curve_comparison = curve_comparison.sort_values(
            ['Segment', 'Cohort', 'Metric', 'MOB']
        ).reset_index(drop=True)
    else:
        curve_comparison = pd.DataFrame()

    # If backtest actuals are available, add post-cutoff actual rate overlays by MOB
    # so users can compare forecast-applied rates vs observed post-cutoff actual rates
    # on the same MOB axis.
    if len(curve_comparison) > 0 and backtest is not None and len(backtest) > 0:
        bt = backtest.copy()
        bt = bt[bt['Metric'].isin(['OpeningGBV'] + flow_metrics)].copy()

        if len(bt) > 0:
            bt_pivot = bt.pivot_table(
                index=['Segment', 'Cohort', 'Month'],
                columns='Metric',
                values='Actual',
                aggfunc='first'
            ).reset_index()

            bt_rate_rows = []
            for metric in flow_metrics:
                if metric not in bt_pivot.columns or 'OpeningGBV' not in bt_pivot.columns:
                    continue

                temp = bt_pivot[['Segment', 'Cohort', 'Month', 'OpeningGBV', metric]].copy()
                temp['Actual_Backtest_Rate'] = temp.apply(
                    lambda r: safe_divide(r[metric], r['OpeningGBV']), axis=1
                )

                # Convert backtest month to MOB for side-by-side curve comparison
                # MOB = months_between(cohort_yyyymm, month_yyyymm)
                month_ts = pd.to_datetime(temp['Month'])
                cohort_int = temp['Cohort'].astype(int)
                cohort_year = (cohort_int // 100).astype(int)
                cohort_month = (cohort_int % 100).astype(int)
                temp['MOB'] = (
                    (month_ts.dt.year - cohort_year) * 12 +
                    (month_ts.dt.month - cohort_month)
                )

                temp['Metric'] = metric
                temp['Backtest_Month'] = month_ts
                bt_rate_rows.append(
                    temp[['Segment', 'Cohort', 'Metric', 'MOB', 'Backtest_Month', 'Actual_Backtest_Rate']]
                )

            if bt_rate_rows:
                bt_rates = pd.concat(bt_rate_rows, ignore_index=True)
                bt_rates.rename(columns={'MOB': 'Backtest_MOB', 'Backtest_Month': 'Forecast_Month'}, inplace=True)
                curve_comparison = curve_comparison.merge(
                    bt_rates,
                    on=['Segment', 'Cohort', 'Metric', 'Forecast_Month'],
                    how='left'
                )

                curve_comparison['Rate_Delta_Forecast_minus_BacktestActual'] = (
                    curve_comparison['Forecast_Applied_Rate'] - curve_comparison['Actual_Backtest_Rate']
                )
                curve_comparison['MOB_Alignment_Diff'] = (
                    curve_comparison['MOB'] - curve_comparison['Backtest_MOB']
                )

                # Keep the historical-delta column as-is and include new backtest columns
                curve_comparison = curve_comparison[
                    ['Segment', 'Cohort', 'Metric', 'MOB',
                     'Actual_Historical_Rate', 'Forecast_Applied_Rate',
                     'Rate_Delta_Forecast_minus_Actual',
                     'Actual_Backtest_Rate', 'Rate_Delta_Forecast_minus_BacktestActual',
                     'Backtest_MOB', 'MOB_Alignment_Diff',
                     'Forecast_Month', 'Forecast_Approach']
                ]

    # Monthly-aligned comparison (one row per Segment x Cohort x Metric x Month)
    # to make pivots/charts easier and avoid MOB-only alignment confusion.
    curve_comparison_monthly = pd.DataFrame()
    if backtest is not None and len(backtest) > 0:
        monthly_rows = []
        for metric in flow_metrics:
            rate_col = f'{metric}_Rate'
            approach_col = f'{metric}_Approach'

            if rate_col not in forecast.columns:
                continue

            fc_cols = ['Segment', 'Cohort', 'MOB', 'ForecastMonth', rate_col]
            if approach_col in forecast.columns:
                fc_cols.append(approach_col)

            fc = forecast[fc_cols].copy()
            fc.rename(columns={
                'ForecastMonth': 'Month',
                'MOB': 'Forecast_MOB',
                rate_col: 'Forecast_Applied_Rate',
            }, inplace=True)
            if approach_col in fc.columns:
                fc.rename(columns={approach_col: 'Forecast_Approach'}, inplace=True)
            else:
                fc['Forecast_Approach'] = ''

            bt = backtest[backtest['Metric'].isin(['OpeningGBV', metric])].copy()
            bt_pivot = bt.pivot_table(
                index=['Segment', 'Cohort', 'Month'],
                columns='Metric',
                values='Actual',
                aggfunc='first'
            ).reset_index()
            if 'OpeningGBV' not in bt_pivot.columns or metric not in bt_pivot.columns:
                continue

            bt_pivot['Actual_Backtest_Rate'] = bt_pivot.apply(
                lambda r: safe_divide(r[metric], r['OpeningGBV']), axis=1
            )
            bt_pivot['Metric'] = metric
            bt_pivot['Month'] = pd.to_datetime(bt_pivot['Month'])

            # Backtest MOB from cohort + month
            cohort_int = bt_pivot['Cohort'].astype(int)
            cohort_year = (cohort_int // 100).astype(int)
            cohort_month = (cohort_int % 100).astype(int)
            bt_pivot['Backtest_MOB'] = (
                (bt_pivot['Month'].dt.year - cohort_year) * 12 +
                (bt_pivot['Month'].dt.month - cohort_month)
            )

            bt_rates = bt_pivot[['Segment', 'Cohort', 'Month', 'Metric', 'Actual_Backtest_Rate', 'Backtest_MOB']]

            merged_monthly = fc.merge(
                bt_rates,
                on=['Segment', 'Cohort', 'Month'],
                how='left'
            )
            merged_monthly['Metric'] = metric
            merged_monthly['Rate_Delta_Forecast_minus_BacktestActual'] = (
                merged_monthly['Forecast_Applied_Rate'] - merged_monthly['Actual_Backtest_Rate']
            )

            monthly_rows.append(
                merged_monthly[
                    ['Segment', 'Cohort', 'Metric', 'Month',
                     'Forecast_MOB', 'Backtest_MOB',
                     'Forecast_Applied_Rate', 'Actual_Backtest_Rate',
                     'Rate_Delta_Forecast_minus_BacktestActual',
                     'Forecast_Approach']
                ]
            )

        if monthly_rows:
            curve_comparison_monthly = pd.concat(monthly_rows, ignore_index=True)
            curve_comparison_monthly = curve_comparison_monthly.sort_values(
                ['Segment', 'Cohort', 'Metric', 'Month']
            ).reset_index(drop=True)

    # ==========================================================================
    # Prepare Seasonal Factors
    # ==========================================================================
    seasonal_factors_df = pd.DataFrame()
    if Config.ENABLE_SEASONALITY:
        seasonal_factors = calculate_seasonal_factors(fact_raw)
        seasonal_rows = []
        for segment, factors in seasonal_factors.items():
            for month, factor in factors.items():
                seasonal_rows.append({
                    'Segment': segment,
                    'Month_Number': month,
                    'Month_Name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1],
                    'Seasonal_Factor': round(factor, 4),
                    'Interpretation': f"CR typically {'higher' if factor > 1 else 'lower'} than average by {abs(factor-1)*100:.1f}%"
                })
        seasonal_factors_df = pd.DataFrame(seasonal_rows)

    # ==========================================================================
    # Write Excel File
    # ==========================================================================
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # README sheet
        sheet_names = [
                '1_Actuals_Data',
                '2_Historical_Rates',
                '3_Extended_Curves',
                '4_Methodology_Applied',
                '5_Forecast_Output',
                '6_Combined_View',
                '7_Rate_Methodology_Rules',
                '8_Seasonal_Factors',
                '9_Summary',
                '10_Details',
                '11_Impairment',
                '12_Reconciliation',
                '13_Validation'
        ]
        descriptions = [
                'Raw historical data with calculated rates for each month',
                'Aggregated rate curves by Segment x Cohort x MOB (historical only)',
                'Rate curves extended for forecast period (Historical + Extended)',
                'Which forecast approach was used for each Segment x Cohort x MOB x Metric',
                'Final forecast output with all calculated amounts',
                'Actuals + Forecast combined for easy comparison and charting',
                'The methodology rules from Rate_Methodology.csv',
                'Seasonal adjustment factors by Segment and Month',
                'Monthly aggregated forecast summary',
                'Full cohort-level forecast details',
                'Impairment and provision analysis',
                'GBV reconciliation checks',
                'Validation rules and pass/fail status'
        ]
        use_for = [
                'Pivot tables showing historical trends, validating raw data',
                'Understanding historical rate patterns by cohort age (MOB)',
                'Seeing how rates are projected forward',
                'Auditing which approach (CohortAvg, Manual, etc.) was used',
                'Final forecast numbers for reporting',
                'Building charts showing Actual vs Forecast over time',
                'Reference for methodology rules',
                'Understanding how monthly seasonality affects CR forecasts',
                'High-level monthly/segment reporting',
                'Detailed cohort-by-cohort analysis',
                'Provision and coverage ratio analysis',
                'Verifying GBV movements reconcile correctly',
                'Quality assurance and data validation'
        ]
        if backtest is not None and len(backtest) > 0:
            sheet_names.append('14_Backtest_Comparison')
            descriptions.append('Forecast vs actuals comparison by Segment x Cohort x Month (backtest)')
            use_for.append('Analysing forecast accuracy against post-cutoff actuals for BB cohorts')
        if ex_contra_actuals is not None and len(ex_contra_actuals) > 0:
            sheet_names.append('15_ExContra_Actuals')
            descriptions.append('Ex-contra actuals roll-forward: GBV series excluding contra settlements, with derived provision and NBV')
            use_for.append('Like-for-like comparison basis vs forecast (which also excludes contra). Includes contra effect sanity check.')
        if len(curve_comparison) > 0:
            sheet_names.append('16_Curve_Comparison')
            descriptions.append('Side-by-side historical actual rates and (where available) post-cutoff backtest actual rates vs forecast-applied rates by Segment x Cohort x MOB for key flow metrics')
            use_for.append('Curve QC: compare historical and post-cutoff observed levels/shapes to forecasted curve shape by cohort')
        if len(curve_comparison_monthly) > 0:
            sheet_names.append('17_Curve_Comparison_Monthly')
            descriptions.append('Month-aligned forecast vs backtest actual rates by Segment x Cohort x Metric, with MOB fields for both sides')
            use_for.append('Pivot/chart-ready month-by-month comparison without MOB-only row mismatches')
        readme_data = {
            'Sheet Name': sheet_names,
            'Description': descriptions,
            'Use For': use_for
        }
        readme_df = pd.DataFrame(readme_data)
        readme_df.to_excel(writer, sheet_name='README', index=False)

        # Transparency sheets
        actuals_output.to_excel(writer, sheet_name='1_Actuals_Data', index=False)
        historical_curves.to_excel(writer, sheet_name='2_Historical_Rates', index=False)
        extended_curves.to_excel(writer, sheet_name='3_Extended_Curves', index=False)
        rate_lookup.to_excel(writer, sheet_name='4_Methodology_Applied', index=False)
        forecast.to_excel(writer, sheet_name='5_Forecast_Output', index=False)
        combined_df.to_excel(writer, sheet_name='6_Combined_View', index=False)
        methodology.to_excel(writer, sheet_name='7_Rate_Methodology_Rules', index=False)

        if len(seasonal_factors_df) > 0:
            seasonal_factors_df.to_excel(writer, sheet_name='8_Seasonal_Factors', index=False)

        # Output sheets
        summary.to_excel(writer, sheet_name='9_Summary', index=False)
        details.to_excel(writer, sheet_name='10_Details', index=False)
        impairment_output.to_excel(writer, sheet_name='11_Impairment', index=False)
        reconciliation.to_excel(writer, sheet_name='12_Reconciliation', index=False)
        validation.to_excel(writer, sheet_name='13_Validation', index=False)

        if backtest is not None and len(backtest) > 0:
            backtest.to_excel(writer, sheet_name='14_Backtest_Comparison', index=False)

        if ex_contra_actuals is not None and len(ex_contra_actuals) > 0:
            ex_contra_actuals.to_excel(writer, sheet_name='15_ExContra_Actuals', index=False)

        if len(curve_comparison) > 0:
            curve_comparison.to_excel(writer, sheet_name='16_Curve_Comparison', index=False)

        if len(curve_comparison_monthly) > 0:
            curve_comparison_monthly.to_excel(writer, sheet_name='17_Curve_Comparison_Monthly', index=False)

        # ── DS Diagnostics sheets ──────────────────────────────────────────────
        # Sheet 18: One-row-per-cohort donor summary (readable at a glance)
        # Sheet 19: Layer B calibration detail (actual vs implied by quarter)
        # Sheet 20: Pre-scale DS rate profiles as event-number pivot
        if ds_donor_map is not None and len(ds_donor_map) > 0:
            # Build a clean 1-row-per-(Segment,Cohort) summary ─────────────────
            donor_summary = (
                ds_donor_map
                .drop_duplicates(subset=['Segment', 'Cohort'])
                [['Segment', 'Cohort', 'DonorCohort', 'IsAutoSelected',
                  'N_DS_Obs_Target', 'N_DS_Obs_Donor', 'ShrinkageAlpha']]
                .rename(columns={
                    'DonorCohort':     'Donor_Cohort',
                    'IsAutoSelected':  'Auto_CR_Selected',
                    'N_DS_Obs_Target': 'N_Own_DS_Events',
                    'N_DS_Obs_Donor':  'N_Donor_DS_Events',
                    'ShrinkageAlpha':  'Shrinkage_Alpha (own weight)',
                })
                .sort_values(['Segment', 'Cohort'])
            )
            donor_summary.to_excel(writer, sheet_name='18_DS_Donor_Map', index=False)

            # Build event-number rate-profile pivot ───────────────────────────
            # Rows = Segment+Cohort, columns = DS event number, values = rate %
            rate_pivot_rows = []
            for seg in ds_donor_map['Segment'].unique():
                for coh in ds_donor_map[ds_donor_map['Segment'] == seg]['Cohort'].unique():
                    mask = (ds_donor_map['Segment'] == seg) & (ds_donor_map['Cohort'] == coh)
                    sub = ds_donor_map[mask].sort_values('DS_Event_Number')
                    row: Dict[str, Any] = {
                        'Segment': seg,
                        'Cohort': coh,
                        'Donor_Cohort': sub['DonorCohort'].iloc[0],
                        'Shrinkage_Alpha': sub['ShrinkageAlpha'].iloc[0],
                    }
                    for _, r in sub.iterrows():
                        ev = int(r['DS_Event_Number'])
                        row[f'E{ev:02d}_Rate_pct'] = round(r['WO_DebtSold_Rate_DS'] * 100, 2)
                    rate_pivot_rows.append(row)
            if rate_pivot_rows:
                rate_pivot_df = (
                    pd.DataFrame(rate_pivot_rows)
                    .sort_values(['Segment', 'Cohort'])
                )
                rate_pivot_df.to_excel(
                    writer, sheet_name='20_DS_Rate_Profiles', index=False
                )

        if ds_scale_factors_diag is not None and len(ds_scale_factors_diag) > 0:
            ds_scale_factors_diag.to_excel(writer, sheet_name='19_DS_ScaleFactors', index=False)

    logger.info(f"Comprehensive report saved to: {output_path}")

    print("\n" + "=" * 70)
    print(f"SUCCESS! Comprehensive report saved to: {output_path}")
    print("=" * 70)
    print("\nSheets included:")
    print("  TRANSPARENCY:")
    print("    - README: Guide to understanding each sheet")
    print("    - 1_Actuals_Data: Raw data with calculated rates")
    print("    - 2_Historical_Rates: Rate curves from historical data")
    print("    - 3_Extended_Curves: Curves extended for forecast")
    print("    - 4_Methodology_Applied: Which approach used for each metric")
    print("    - 5_Forecast_Output: Full forecast output")
    print("    - 6_Combined_View: Actuals + Forecast for charting")
    print("    - 7_Rate_Methodology_Rules: Your methodology rules")
    print("    - 8_Seasonal_Factors: Monthly adjustment factors")
    print("  OUTPUTS:")
    print("    - 9_Summary: Monthly aggregated summary")
    print("    - 10_Details: Cohort-level detail")
    print("    - 11_Impairment: Impairment analysis")
    print("    - 12_Reconciliation: GBV reconciliation")
    print("    - 13_Validation: Validation checks")
    if backtest is not None and len(backtest) > 0:
        print("  BACKTEST:")
        print("    - 14_Backtest_Comparison: Forecast vs actuals by Segment x Cohort x Month")
    if ex_contra_actuals is not None and len(ex_contra_actuals) > 0:
        print("    - 15_ExContra_Actuals: Ex-contra actuals roll-forward (GBV, provision, NBV)")
    if len(curve_comparison) > 0:
        print("    - 16_Curve_Comparison: Actual historical rates vs forecast-applied rates by cohort/MOB")
    if len(curve_comparison_monthly) > 0:
        print("    - 17_Curve_Comparison_Monthly: Forecast vs backtest actual rates aligned by month")
    if ds_donor_map is not None and len(ds_donor_map) > 0:
        print("  DS DIAGNOSTICS (DSDonorCRScaled):")
        print("    - 18_DS_Donor_Map: 1 row per cohort — donor assigned, shrinkage alpha, own/donor obs count")
        print("    - 20_DS_Rate_Profiles: Pre-scale DS rate % by event number (cohort × event pivot per segment)")
    if ds_scale_factors_diag is not None and len(ds_scale_factors_diag) > 0:
        print("    - 19_DS_ScaleFactors: Layer B calibration — actual vs implied DS per segment/quarter")

    return output_path


# =============================================================================
# SECTION 13: MAIN ORCHESTRATION
# =============================================================================

def run_backbook_forecast(fact_raw_path: str, methodology_path: str,
                          debt_sale_path: Optional[str], output_dir: str,
                          max_months: int, transparency_report: bool = False,
                          cutoff_date: Optional[str] = None,
                          use_ex_contra: bool = False) -> pd.DataFrame:
    """
    Orchestrate entire forecast process.

    Args:
        fact_raw_path: Path to Fact_Raw_Full.csv
        methodology_path: Path to Rate_Methodology.csv
        debt_sale_path: Path to Debt_Sale_Schedule.csv or None
        output_dir: Output directory
        max_months: Forecast horizon
        transparency_report: If True, generate single comprehensive output file
        cutoff_date: Optional forecast cutoff in YYYY-MM format (e.g., '2025-10').
                     Data before this month is used for curves/seeds; this month
                     becomes the first forecast month. Enables backtest comparison.
        use_ex_contra: If True, use ex-contra actuals for backtest stock metrics.

    Returns:
        pd.DataFrame: Complete forecast
    """
    logger.info("=" * 60)
    logger.info("Starting Backbook Forecast")
    logger.info("=" * 60)

    start_time = datetime.now()

    try:
        # 1. Load data
        logger.info("\n[Step 1/9] Loading data...")
        fact_raw = load_fact_raw(fact_raw_path)
        methodology = load_rate_methodology(methodology_path)
        debt_sale_schedule = load_debt_sale_schedule(debt_sale_path)

        # 1a. Apply cutoff date filtering if specified
        fact_raw_full = None  # Will hold unfiltered data for backtest comparison
        if cutoff_date:
            cutoff_dt = end_of_month(pd.Timestamp(cutoff_date + '-01'))
            cutoff_yyyymm = int(cutoff_date.replace('-', ''))
            logger.info(f"Cutoff date specified: {cutoff_date}")
            logger.info(f"  First forecast month: {cutoff_dt}")
            logger.info(f"  Last actuals month: {end_of_month(cutoff_dt - relativedelta(months=1))}")
            logger.info(f"  Excluding cohorts with origination >= {cutoff_yyyymm}")

            # Store full data for backtest comparison later
            fact_raw_full = fact_raw.copy()

            rows_before = len(fact_raw)
            cohorts_before = fact_raw['Cohort'].nunique()

            # Filter to data before the cutoff month (last actuals = cutoff - 1 month)
            fact_raw = fact_raw[fact_raw['CalendarMonth'] < cutoff_dt].copy()

            # Exclude cohorts that wouldn't exist at the cutoff date
            fact_raw = fact_raw[fact_raw['Cohort'].astype(int) < cutoff_yyyymm].copy()

            rows_after = len(fact_raw)
            cohorts_after = fact_raw['Cohort'].nunique()
            logger.info(f"  Filtered: {rows_before} -> {rows_after} rows, "
                        f"{cohorts_before} -> {cohorts_after} cohorts")

            if rows_after == 0:
                raise ValueError(f"No data remaining after cutoff filter ({cutoff_date}). "
                                 "Check that the cutoff date is within the data range.")

        # 1b. Calculate seasonal factors from historical data
        if Config.ENABLE_SEASONALITY:
            logger.info("\n[Step 1b/9] Calculating seasonal factors...")
            calculate_seasonal_factors(fact_raw)
        else:
            logger.info("\n[Step 1b/9] Seasonality disabled - skipping seasonal factor calculation")

        # 1c. Load overlay adjustments
        if Config.ENABLE_OVERLAYS:
            logger.info("\n[Step 1c/9] Loading overlay adjustments...")
            load_overlays()
        else:
            logger.info("\n[Step 1c/9] Overlays disabled - skipping overlay loading")

        # 2. Calculate curves
        logger.info("\n[Step 2/9] Calculating curves...")
        curves_base = calculate_curves_base(fact_raw)
        curves_extended = extend_curves(curves_base, max_months)

        # 3. Calculate impairment curves
        logger.info("\n[Step 3/9] Calculating impairment curves...")
        impairment_actuals = calculate_impairment_actuals(fact_raw)
        impairment_curves = calculate_impairment_curves(impairment_actuals)

        # 4. Generate seeds
        logger.info("\n[Step 4/9] Generating seeds...")
        seed = generate_seed_curves(fact_raw)

        # 4b. Build DS donor curves and calibrate Layer B scaling (DSDonorCRScaled)
        # These run for all segments regardless of whether DSDonorCRScaled is active in
        # the methodology CSV.  The resulting diagnostics are always written to the
        # transparency report (sheets 18 & 19).  The ds_donor_curves DataFrame is only
        # used by apply_approach() when the Approach column is 'DSDonorCRScaled'.
        logger.info("\n[Step 4b/9] Building DS donor curves and calibrating Layer B scaling...")
        ds_event_df = build_ds_event_history(fact_raw)
        ds_donor_curves = build_ds_donor_curves(ds_event_df, methodology, curves_base)
        ds_scale_factors, ds_scale_factors_diag = calibrate_ds_scaling(
            fact_raw, ds_event_df, ds_donor_curves
        )
        # Merge per-segment scaling factors into a rate_lookup column so run_one_step
        # can read them without needing a new parameter.
        # The column is added AFTER build_rate_lookup returns, then read in run_one_step.

        # 5. Build rate lookups
        logger.info("\n[Step 5/9] Building rate lookups...")
        rate_lookup = build_rate_lookup(
            seed, curves_extended, methodology, max_months,
            ds_donor_curves=ds_donor_curves,
        )
        # Attach per-segment DS scaling factors so run_one_step can apply Layer B.
        # Non-DSDonorCRScaled segments receive 1.0 (no change to existing behaviour).
        rate_lookup['WO_DebtSold_ScaleFactor'] = (
            rate_lookup['Segment'].map(ds_scale_factors).fillna(1.0)
        )

        # Use curves_base for coverage ratio lookups instead of impairment_curves.
        # impairment_curves computes coverage ratios by aggregating across CalendarMonth
        # first (mixing different MOBs together when a cohort has sub-groups), which
        # produces incorrect per-MOB ratios. curves_base groups by MOB directly, giving
        # the correct per-MOB coverage ratios needed by SegMedian/CohortAvg/etc.
        impairment_lookup = build_impairment_lookup(
            seed, curves_base, methodology, max_months, debt_sale_schedule
        )

        # 6. Run forecast
        logger.info("\n[Step 6/9] Running forecast...")
        forecast = run_forecast(seed, rate_lookup, impairment_lookup, max_months)

        if len(forecast) == 0:
            logger.error("No forecast data generated")
            return pd.DataFrame()

        # 7. Generate outputs
        logger.info("\n[Step 7/10] Generating outputs...")
        summary = generate_summary_output(forecast)
        details = generate_details_output(forecast)
        impairment_output = generate_impairment_output(forecast)
        reconciliation, validation = generate_validation_output(forecast)

        # 8. Generate backtest comparison and ex-contra actuals if cutoff was specified
        backtest = None
        ex_contra_actuals = None
        if cutoff_date and fact_raw_full is not None:
            forecast_cohorts = forecast[['Segment', 'Cohort']].drop_duplicates()

            # Build ex-contra actuals series (always, for the detail sheet)
            logger.info("\n[Step 8a/10] Building ex-contra actuals series...")
            ex_contra_actuals = build_ex_contra_actuals(fact_raw_full, cutoff_date, forecast_cohorts)
            if len(ex_contra_actuals) == 0:
                ex_contra_actuals = None

            # Generate backtest comparison
            logger.info("\n[Step 8b/10] Generating backtest comparison...")
            backtest = generate_backtest_comparison(
                fact_raw_full, forecast, cutoff_date, use_ex_contra=use_ex_contra
            )
            if len(backtest) == 0:
                logger.warning("  No backtest data generated (no post-cutoff actuals found)")
                backtest = None

        # 9. Export to Excel
        logger.info("\n[Step 9/10] Exporting to Excel...")

        if transparency_report:
            # Generate single comprehensive transparency report (includes backtest if available)
            generate_comprehensive_transparency_report(
                fact_raw=fact_raw,
                methodology=methodology,
                curves_base=curves_base,
                curves_extended=curves_extended,
                rate_lookup=rate_lookup,
                impairment_lookup=impairment_lookup,
                forecast=forecast,
                summary=summary,
                details=details,
                impairment_output=impairment_output,
                reconciliation=reconciliation,
                validation=validation,
                output_dir=output_dir,
                max_months=max_months,
                backtest=backtest,
                ex_contra_actuals=ex_contra_actuals,
                ds_donor_map=ds_donor_curves if len(ds_donor_curves) > 0 else None,
                ds_scale_factors_diag=ds_scale_factors_diag if len(ds_scale_factors_diag) > 0 else None,
            )
        else:
            # Generate separate output files
            export_to_excel(summary, details, impairment_output, reconciliation, validation, output_dir)

            # Generate combined actuals + forecast for variance analysis
            logger.info("\n[Step 9b/10] Generating combined actuals + forecast output...")
            combined = generate_combined_actuals_forecast(fact_raw, forecast, output_dir)

            # Write standalone backtest file if available
            if backtest is not None or ex_contra_actuals is not None:
                os.makedirs(output_dir, exist_ok=True)
                backtest_path = os.path.join(output_dir, 'Backtest_Comparison.xlsx')
                with pd.ExcelWriter(backtest_path, engine='openpyxl') as writer:
                    if backtest is not None:
                        backtest.to_excel(writer, sheet_name='Backtest_Detail', index=False)
                    if ex_contra_actuals is not None:
                        ex_contra_actuals.to_excel(writer, sheet_name='ExContra_Actuals', index=False)
                logger.info(f"  Backtest comparison saved to: {backtest_path}")

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info(f"Forecast complete in {elapsed:.2f} seconds")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)

        # Print validation summary
        if len(validation) > 0:
            overall = validation[validation['Check'] == 'Overall'].iloc[0]
            logger.info(f"\nValidation Summary: {overall['Status']}")
            logger.info(f"  Total checks: {overall['Total_Rows']}")
            logger.info(f"  Passed: {overall['Passed']}")
            logger.info(f"  Failed: {overall['Failed']}")

        return forecast

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Backbook Forecasting Model - Calculate loan portfolio performance forecasts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backbook_forecast.py --fact-raw Fact_Raw_Full.csv --methodology Rate_Methodology.csv
  python backbook_forecast.py --fact-raw data/Fact_Raw_Full.csv --methodology data/Rate_Methodology.csv --months 24 --output results/
        """
    )

    parser.add_argument(
        '--fact-raw', '-f',
        required=True,
        help='Path to Fact_Raw_Full.csv (historical loan data)'
    )

    parser.add_argument(
        '--methodology', '-m',
        required=True,
        help='Path to Rate_Methodology.csv (rate calculation rules)'
    )

    parser.add_argument(
        '--debt-sale', '-d',
        required=False,
        default=None,
        help='Path to Debt_Sale_Schedule.csv (optional debt sale assumptions)'
    )

    parser.add_argument(
        '--output', '-o',
        required=False,
        default='output',
        help='Output directory (default: output/)'
    )

    parser.add_argument(
        '--months', '-n',
        required=False,
        type=int,
        default=12,
        help='Forecast horizon in months (default: 12)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--transparency-report', '-t',
        action='store_true',
        help='Generate single comprehensive Forecast_Transparency_Report.xlsx with all outputs'
    )

    parser.add_argument(
        '--cutoff', '-c',
        required=False,
        default=None,
        help='Forecast cutoff date in YYYY-MM format (e.g., 2025-10). '
             'Data before this month is used for curves/seeds; this month becomes the first forecast month. '
             'If omitted, uses all available data (default behaviour). '
             'When set, auto-generates a Backtest_Comparison sheet comparing forecast vs post-cutoff actuals.'
    )

    parser.add_argument(
        '--ex-contra',
        action='store_true',
        help='Use ex-contra actuals for backtest comparison. '
             'Builds a month-by-month GBV roll-forward that excludes contra settlements '
             '(matching the forecast bridge), so stock metrics (GBV, provision, NBV) '
             'are compared on a like-for-like basis. Requires --cutoff.'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_backbook_forecast(
        fact_raw_path=args.fact_raw,
        methodology_path=args.methodology,
        debt_sale_path=args.debt_sale,
        output_dir=args.output,
        max_months=args.months,
        transparency_report=args.transparency_report,
        cutoff_date=args.cutoff,
        use_ex_contra=args.ex_contra
    )


if __name__ == '__main__':
    main()