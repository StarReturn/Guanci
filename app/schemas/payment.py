from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


OnlinePaymentChannel = Literal["wechat", "apple", "alipay"]
PaymentEnvironment = Literal["sandbox", "production"]


class PaymentChannelConfigWrite(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=64)
    environment: PaymentEnvironment
    public_config: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)


class PaymentChannelConfigRead(BaseModel):
    config_id: str
    channel: OnlinePaymentChannel
    display_name: str
    environment: PaymentEnvironment
    status: str
    public_config: dict[str, Any]
    secret_fields: dict[str, str]
    config_version: int
    serving_config_version: int | None
    rotation_fallback_valid_until: datetime | None
    validation_status: str
    validation_message: str | None
    validated_at: datetime | None
    activated_at: datetime | None
    updated_by: str | None
    created_at: datetime
    updated_at: datetime


class PaymentChannelConfigListResponse(BaseModel):
    items: list[PaymentChannelConfigRead]
    total: int
    hot_reload_max_delay_seconds: int = 5


class PaymentChannelValidationRead(BaseModel):
    channel: OnlinePaymentChannel
    config_version: int
    valid: bool
    message: str


class PaymentStartRead(BaseModel):
    channel: OnlinePaymentChannel
    order_no: str
    order_status: str
    config_version: int
    payload: dict[str, Any]


class AppleTransactionRestoreRequest(BaseModel):
    signed_transaction: str = Field(..., min_length=64, max_length=200000)


class PaymentWebhookAck(BaseModel):
    accepted: bool = True
    duplicate: bool = False
    event_id: str | None = None


class PaymentEventRead(BaseModel):
    event_id: str
    channel: str
    channel_event_id: str
    event_type: str
    signature_verified: bool
    payload_digest: str
    config_version: int | None
    processing_status: str
    related_order_id: str | None
    received_at: datetime
    processing_started_at: datetime | None
    processed_at: datetime | None
    next_retry_at: datetime | None
    error_message: str | None
    retry_count: int


class PaymentEventListResponse(BaseModel):
    items: list[PaymentEventRead]
    total: int


class PaymentRefundExecutionRead(BaseModel):
    refund_no: str
    order_no: str
    channel: str
    status: str
    channel_refund_id: str | None
    message: str | None = None


class PaymentReconciliationCreate(BaseModel):
    channel: OnlinePaymentChannel
    from_at: datetime
    to_at: datetime
    limit: int = Field(default=100, ge=1, le=500)


class PaymentReconciliationRead(BaseModel):
    run_id: str
    channel: str
    from_at: datetime
    to_at: datetime
    status: str
    checked_count: int
    mismatch_count: int
    report: dict[str, Any]
    requested_by: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class PaymentReconciliationListResponse(BaseModel):
    items: list[PaymentReconciliationRead]
    total: int


__all__ = [
    "AppleTransactionRestoreRequest",
    "OnlinePaymentChannel",
    "PaymentChannelConfigListResponse",
    "PaymentChannelConfigRead",
    "PaymentChannelConfigWrite",
    "PaymentChannelValidationRead",
    "PaymentEnvironment",
    "PaymentEventListResponse",
    "PaymentEventRead",
    "PaymentReconciliationCreate",
    "PaymentReconciliationListResponse",
    "PaymentReconciliationRead",
    "PaymentRefundExecutionRead",
    "PaymentStartRead",
    "PaymentWebhookAck",
]
