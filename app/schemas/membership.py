from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


BillingCycle = Literal["month", "quarter", "year"]
PaymentChannel = Literal["manual", "wechat", "apple", "alipay"]


class MembershipPriceWrite(BaseModel):
    billing_cycle: BillingCycle
    duration_months: int = Field(..., ge=1, le=36)
    currency: str = Field(default="CNY", min_length=3, max_length=8)
    amount_cents: int = Field(..., ge=0)
    channel: PaymentChannel = "manual"
    external_product_id: str | None = Field(default=None, max_length=256)
    auto_renew: bool = True


class MembershipPriceRead(MembershipPriceWrite):
    price_id: str
    status: str


class MembershipPlanVersionCreate(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=64)
    subtitle: str | None = Field(default=None, max_length=256)
    badge: str | None = Field(default=None, max_length=64)
    entitlements: dict[str, Any]
    prices: list[MembershipPriceWrite]
    change_summary: str | None = Field(default=None, max_length=4000)


class MembershipPlanVersionRead(BaseModel):
    plan_version_id: str
    plan_id: str
    version_no: int
    display_name: str
    subtitle: str | None
    badge: str | None
    entitlement_schema_version: int
    entitlements: dict[str, Any]
    status: str
    effective_at: datetime | None
    published_at: datetime | None
    published_by: str | None
    change_summary: str | None
    prices: list[MembershipPriceRead] = Field(default_factory=list)
    created_at: datetime


class MembershipPlanCreate(BaseModel):
    code: str = Field(..., min_length=2, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    tier_rank: int = Field(..., ge=0, le=10000)
    sort_order: int = Field(default=0, ge=0, le=10000)
    version: MembershipPlanVersionCreate


class MembershipPlanMetadataUpdate(BaseModel):
    tier_rank: int | None = Field(default=None, ge=0, le=10000)
    sort_order: int | None = Field(default=None, ge=0, le=10000)
    status: Literal["active", "inactive"] | None = None


class MembershipPlanRead(BaseModel):
    plan_id: str
    code: str
    tier_rank: int
    status: str
    sort_order: int
    current_version: MembershipPlanVersionRead | None
    created_at: datetime
    updated_at: datetime


class MembershipPlanListResponse(BaseModel):
    items: list[MembershipPlanRead]
    total: int


class VersionValidationRead(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class VersionScheduleRequest(BaseModel):
    effective_at: datetime


class VersionImpactRead(BaseModel):
    current_version_id: str | None
    target_version_id: str
    active_subscription_count: int
    active_period_count: int
    changed_entitlements: dict[str, dict[str, Any]]
    changed_prices: dict[str, dict[str, Any]]
    renewal_affected_subscription_count: int
    auto_renew_disabled_subscription_count: int
    affects_existing_periods: bool = False


class MembershipOrderCreate(BaseModel):
    plan_code: str = Field(..., min_length=2, max_length=32)
    billing_cycle: BillingCycle
    channel: PaymentChannel
    idempotency_key: str = Field(..., min_length=8, max_length=128)
    order_type: Literal["new", "renewal", "change"] = "new"


class MembershipOrderRead(BaseModel):
    order_id: str
    order_no: str
    user_id: str
    plan_code: str
    plan_version_id: str
    billing_cycle: str
    payment_channel: str
    amount_cents: int
    refunded_amount_cents: int
    currency: str
    status: str
    order_type: str
    paid_at: datetime | None
    closed_at: datetime | None
    created_at: datetime


class MembershipOrderListResponse(BaseModel):
    items: list[MembershipOrderRead]
    total: int


class EffectiveMembershipRead(BaseModel):
    source: Literal["subscription", "free"]
    subscription_id: str | None
    period_id: str | None
    plan_code: str
    plan_version_id: str
    version_no: int
    display_name: str
    billing_cycle: str | None
    period_start: datetime
    period_end: datetime
    auto_renew: bool
    cancel_at_period_end: bool
    entitlements: dict[str, Any]


class EntitlementUsageItem(BaseModel):
    entitlement_key: str
    limit: int | bool | str | list[Any]
    reserved: int = 0
    confirmed: int = 0
    remaining: int | None = None
    window_start: datetime
    window_end: datetime


class EntitlementUsageRead(BaseModel):
    membership: EffectiveMembershipRead
    items: list[EntitlementUsageItem]


class SubscriptionRead(BaseModel):
    subscription_id: str
    user_id: str
    plan_code: str
    status: str
    payment_channel: str
    auto_renew: bool
    cancel_at_period_end: bool
    current_period_start: datetime | None
    current_period_end: datetime | None
    created_at: datetime


class SubscriptionListResponse(BaseModel):
    items: list[SubscriptionRead]
    total: int


class ChangePlanRequest(BaseModel):
    plan_code: str = Field(..., min_length=2, max_length=32)
    billing_cycle: BillingCycle
    channel: PaymentChannel
    idempotency_key: str = Field(..., min_length=8, max_length=128)


class AdminGrantSubscriptionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    plan_code: str = Field(..., min_length=2, max_length=32)
    billing_cycle: BillingCycle
    duration_months: int | None = Field(default=None, ge=1, le=120)
    reason: str = Field(..., min_length=2, max_length=500)
    replace_active: bool = False

    @model_validator(mode="after")
    def validate_duration(self):
        if self.duration_months is not None and self.duration_months <= 0:
            raise ValueError("duration_months must be positive")
        return self


class AdminRevokeSubscriptionRequest(BaseModel):
    reason: str = Field(..., min_length=2, max_length=500)


class AdminActivateOrderRequest(BaseModel):
    transaction_id: str | None = Field(default=None, max_length=256)
