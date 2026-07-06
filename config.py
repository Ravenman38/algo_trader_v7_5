"""
AlgoTrader v4 configuration.
Change these values here first before editing strategy code.
"""

# Portfolio / execution
PORTFOLIO_VALUE = 100_000.0
INITIAL_CAPITAL = 100_000.0

# Risk and sizing
TARGET_RISK_PCT = 0.005
MAX_POSITION_PCT = 0.15
MAX_DEPLOY_PCT = 1.00
MIN_POSITION_PCT = 0.02

# IBKR-style commission estimate
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00

# Realism knobs for future execution-model testing
SLIPPAGE_BPS = 0.0
EXECUTION_MODE = "current_close"  # future option: next_open

# Screener / order filters
MIN_PROB = 0.30
MIN_SIGNALS = 2
MIN_SAR_GAP = 0.0

# Backtest / reporting
BACKTEST_START_DATE = "2018-01-01"
BACKTEST_YEARS = 8
RESCREEN_DAYS = 5
HOLDING_DAYS = 5
MAX_TICKERS = 900
USE_STOPS = True
BENCHMARK = "SPY"
RANDOM_SEED = 42
