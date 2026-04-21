from pydantic import BaseModel, Field
from typing import Optional

class AnalystSignal(BaseModel):
    signal: str = Field(description="Must be BUY, SELL, or HOLD")
    confidence: float = Field(description="Score between 0 and 1")
    strategy_used: str = Field(description="Which strategy snippet was applied")
    reasoning: str = Field(description="Brief technical justification")
    internal_monologue: str = Field(description="The full step-by-step thought process")
    entry_condition : str = Field(description = "The market conditions when to trade ")
    exit_condition : str = Field(description = "The market conditions for when to exit the trade ")
    valid_till : str = Field(description="Duration till when to keep searching for entry condition")
    asset_name:str = Field(description="Name")

class RiskAssessment(BaseModel):
    is_approved: bool = Field(description="Whether the trade meets safety criteria")
    final_position_size: float = Field(description="The actual amount to trade (0.0 if rejected)")
    stop_loss_price: Optional[float] = Field(description="Calculated price to exit at a loss")
    take_profit_price: Optional[float] = Field(description="Calculated price to exit at a profit")
    risk_score: int = Field(description="1-10 safety rating")
    risk_monologue: str = Field(description="Reasoning for the approval/rejection")