from pprint import pprint

json = [
  {
    "metadata": {
      "name": "LinReg Channel + TSI_MACD Mean Reversion",
      "regime": "Trending"
    },
    "theory": {
      "principle": "This strategy exploits the natural elastic boundaries of a linear regression channel within a broader trend. It waits for the price to reach the outer statistical limits (top/bottom of the channel) while confirming momentum exhaustion using a double-smoothed TSI_MACD crossover, ensuring the elastic snap-back is mathematically probable before execution.",
      "market_inefficiency": "Statistical overextension within a macroeconomic trend, capitalizing on delayed momentum shifts at statistical extremes.",
      "synergy_markers": {
        "works_well_with": [
          "Trend Confirmers",
          "Macro-trend baselines (e.g., 200 EMA)"
        ],
        "conflicts_with": [
          "Counter-trend logic",
          "Sideways chop without an overarching slope"
        ]
      }
    },
    "mechanics": {
      "logic_gate": {
        "entry_primary": [
          "Price touches the bottom or top boundary of the Linear Regression channel."
        ],
        "entry_confirmation": [
          "Price is trading in the direction of the 200 EMA (above for long, below for short).",
          "TSI_MACD indicator crosses in the direction of the mean."
        ]
      },
      "exit_logic": {
        "take_profit": "Price touches the opposite boundary of the Linear Regression channel.",
        "exit_condition": "TSI_MACD crosses in the opposite direction, indicating momentum failure."
      }
    },
    "operational_meta": {
      "risk_profile": "Conservative",
      "dependency_indicators": [
        "Linear Regression Channel",
        "TSI_MACD",
        "200 EMA"
      ]
    }
  },
  {
    "metadata": {
      "name": "Keltner Channels + RSI Scalping",
      "regime": "Trending"
    },
    "theory": {
      "principle": "The strategy's DNA relies on the elastic nature of asset prices around a macroeconomic baseline. It uses Keltner Channels to identify extreme volatility deviations and waits for the fast 7-period RSI momentum oscillator to confirm that the snapped-band effect has actually begun, protecting traders from stepping in front of a runaway trend.",
      "market_inefficiency": "Emotional overreactions (fear/greed) that cause temporary price dislocations from a longer-term moving average.",
      "synergy_markers": {
        "works_well_with": [
          "Trend Filters",
          "Momentum Oscillators"
        ],
        "conflicts_with": [
          "Strong breakout conditions",
          "Range-bound markets lacking a clear 200 EMA slope"
        ]
      }
    },
    "mechanics": {
      "logic_gate": {
        "entry_primary": [
          "RSI exits the oversold zone (crosses above 30) for longs, or exits the overbought zone (crosses below 70) for shorts."
        ],
        "entry_confirmation": [
          "Price is appropriately above/below the 200 EMA trend filter.",
          "Price touches or breaches the outer Keltner Channel bands before the RSI crosses."
        ]
      },
      "exit_logic": {
        "take_profit": "Price touches the basis (middle) line of the Keltner Channel.",
        "exit_condition": "RSI enters the opposite extreme (overbought for longs, oversold for shorts)."
      }
    },
    "operational_meta": {
      "risk_profile": "Conservative",
      "dependency_indicators": [
        "Keltner Channels (Range style)",
        "RSI (7-period)",
        "200 EMA"
      ]
    }
  },
  {
    "metadata": {
      "name": "Bollinger Bands + RSI Extremes",
      "regime": "Sideways"
    },
    "theory": {
      "principle": "Operates on the premise that in a non-trending, range-bound environment, prices will inevitably bounce between the upper and lower statistical extremes defined by two standard deviations. The ultra-fast 5-period RSI provides a hyper-sensitive trigger for the snap-back, while the ADX acts as a strict gatekeeper to avoid trending traps.",
      "market_inefficiency": "Range-bound liquidity hunting, exploiting the lack of directional momentum where price predictably oscillates between standard deviation boundaries.",
      "synergy_markers": {
        "works_well_with": [
          "Volatility Filters (ADX < 25)",
          "Support/Resistance grids"
        ],
        "conflicts_with": [
          "Trending Markets",
          "Momentum Breakouts"
        ]
      }
    },
    "mechanics": {
      "logic_gate": {
        "entry_primary": [
          "Price crosses the outer Bollinger Band."
        ],
        "entry_confirmation": [
          "ADX is below 25 (confirming a non-trending environment).",
          "RSI drops below 30 or above 70, then strictly exits the extreme zone."
        ]
      },
      "exit_logic": {
        "take_profit": "Price touches the opposite Bollinger Band.",
        "exit_condition": "Stop loss triggered at the recent swing extreme just beyond the entry."
      }
    },
    "operational_meta": {
      "risk_profile": "Aggressive",
      "dependency_indicators": [
        "Bollinger Bands (40-period, 2 SD)",
        "RSI (5-period)",
        "ADX (14-period)"
      ]
    }
  },
  {
    "metadata": {
      "name": "VWAP and Stochastic Oscillator Micro-Reversion",
      "regime": "Volatile"
    },
    "theory": {
      "principle": "Designed for micro-timeframes (e.g., 5-minute), this strategy anchors itself to the VWAP, an institutional volume-based mean. It catches micro-reversions when retail traders push the price beyond 1.5 standard deviations, utilizing the Stochastic Oscillator to optimally time the exhaustion of these irrational micro-pushes.",
      "market_inefficiency": "Intraday retail overextensions against institutional volume averages causing delayed reactions to volume spikes.",
      "synergy_markers": {
        "works_well_with": [
          "Volume-weighted means",
          "Tight Volatility Bands"
        ],
        "conflicts_with": [
          "Low volume/illiquid environments",
          "Long-term swing trading"
        ]
      }
    },
    "mechanics": {
      "logic_gate": {
        "entry_primary": [
          "A candle closes back inside the Bollinger Band after previously breaching it."
        ],
        "entry_confirmation": [
          "Price is aligned correctly above/below the VWAP trend.",
          "Stochastic Oscillator confirms extreme exhaustion (below 20 or above 80)."
        ]
      },
      "exit_logic": {
        "take_profit": "Price touches the opposite Bollinger Band.",
        "exit_condition": "Stochastic Oscillator enters the opposite extreme zone."
      }
    },
    "operational_meta": {
      "risk_profile": "Aggressive",
      "dependency_indicators": [
        "VWAP",
        "Bollinger Bands (1.5 SD)",
        "Stochastic Oscillator (7,4,3)"
      ]
    }
  },
  {
    "metadata": {
      "name": "False Breakout (Expo) Liquidity Trap",
      "regime": "Sideways"
    },
    "theory": {
      "principle": "Exploits 'bull traps' and 'bear traps' where price momentarily breaches established support/resistance to capture liquidity (triggering retail stop-losses), only to sharply reverse direction back into the established range.",
      "market_inefficiency": "Liquidity gaps and trap patterns at key structural boundaries, exploiting the mechanical triggering of stop-loss clusters.",
      "synergy_markers": {
        "works_well_with": [
          "Support/Resistance Zones",
          "Aggressive Signal Settings"
        ],
        "conflicts_with": [
          "High-momentum breakouts",
          "Strong trend continuations"
        ]
      }
    },
    "mechanics": {
      "logic_gate": {
        "entry_primary": [
          "A false break of a support or resistance level occurs."
        ],
        "entry_confirmation": [
          "The False Breakout (Expo) indicator prints a green or red confirmation triangle."
        ]
      },
      "exit_logic": {
        "take_profit": "Price reverts to the opposite end of the consolidation range.",
        "exit_condition": "An opposite false breakout signal appears."
      }
    },
    "operational_meta": {
      "risk_profile": "Aggressive",
      "dependency_indicators": [
        "False Breakout (Expo) Indicator",
        "Horizontal Support/Resistance"
      ]
    }
  }
]

pprint(json)