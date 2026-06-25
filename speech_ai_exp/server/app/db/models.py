from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    preferred_name: Mapped[str] = mapped_column(String(64))
    full_name: Mapped[str] = mapped_column(String(128))
    email: Mapped[str] = mapped_column(String(256), index=True)
    phone: Mapped[str] = mapped_column(String(32), default="")
    account_status: Mapped[str] = mapped_column(String(32), default="Active")
    loyalty_program: Mapped[str] = mapped_column(String(64), default="")
    loyalty_tier: Mapped[str] = mapped_column(String(32), default="")
    loyalty_points: Mapped[int] = mapped_column(Integer, default=0)
    store_credit_usd: Mapped[str] = mapped_column(String(32), default="0.00")
    default_address: Mapped[str] = mapped_column(Text, default="")
    alternate_address: Mapped[str] = mapped_column(Text, default="")
    time_zone: Mapped[str] = mapped_column(String(64), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    orders: Mapped[list[Order]] = relationship(back_populates="customer")
    invoices: Mapped[list[Invoice]] = relationship(back_populates="customer")
    payments: Mapped[list[Payment]] = relationship(back_populates="customer")
    licenses: Mapped[list[License]] = relationship(back_populates="customer")
    subscriptions: Mapped[list[Subscription]] = relationship(back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    display_order_number: Mapped[str] = mapped_column(String(32), default="", index=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    order_date: Mapped[str] = mapped_column(String(32))
    channel: Mapped[str] = mapped_column(String(32), default="")
    order_status: Mapped[str] = mapped_column(String(32), index=True)
    payment_status: Mapped[str] = mapped_column(String(32), default="")
    fulfillment_status: Mapped[str] = mapped_column(String(32), default="")
    shipping_address: Mapped[str] = mapped_column(Text, default="")
    subtotal: Mapped[str] = mapped_column(String(32), default="")
    discount: Mapped[str] = mapped_column(String(32), default="")
    shipping_fee: Mapped[str] = mapped_column(String(32), default="")
    sales_tax: Mapped[str] = mapped_column(String(32), default="")
    order_total: Mapped[str] = mapped_column(String(32), default="")
    promotion_code: Mapped[str] = mapped_column(String(32), default="")
    customer_note: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(default=False)
    is_open: Mapped[bool] = mapped_column(default=False)
    is_delayed: Mapped[bool] = mapped_column(default=False)
    delay_reason: Mapped[str] = mapped_column(Text, default="")
    cancellation_penalty_usd: Mapped[str] = mapped_column(String(32), default="0.00")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    customer: Mapped[Customer] = relationship(back_populates="orders")
    line_items: Mapped[list[LineItem]] = relationship(back_populates="order", cascade="all, delete-orphan")
    shipments: Mapped[list[Shipment]] = relationship(back_populates="order", cascade="all, delete-orphan")


class LineItem(Base):
    __tablename__ = "line_items"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    sku: Mapped[str] = mapped_column(String(32), index=True)
    product_name: Mapped[str] = mapped_column(String(256))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price: Mapped[str] = mapped_column(String(32), default="")
    net_amount: Mapped[str] = mapped_column(String(32), default="")
    fulfillment_status: Mapped[str] = mapped_column(String(32), default="")
    shipment_id: Mapped[str] = mapped_column(String(32), default="")
    serial_number: Mapped[str] = mapped_column(String(64), default="")
    warranty: Mapped[str] = mapped_column(String(128), default="")
    return_window_end: Mapped[str] = mapped_column(String(32), default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    order: Mapped[Order] = relationship(back_populates="line_items")


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    carrier: Mapped[str] = mapped_column(String(32), default="")
    tracking_number: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(32), index=True)
    service: Mapped[str] = mapped_column(String(64), default="")
    ship_date: Mapped[str] = mapped_column(String(32), default="")
    estimated_delivery_date: Mapped[str] = mapped_column(String(32), default="")
    delivery_date: Mapped[str] = mapped_column(String(32), default="")
    delivery_location: Mapped[str] = mapped_column(String(128), default="")
    signature_required: Mapped[str] = mapped_column(String(8), default="")
    contents: Mapped[str] = mapped_column(Text, default="")

    order: Mapped[Order] = relationship(back_populates="shipments")
    tracking_events: Mapped[list[TrackingEvent]] = relationship(
        back_populates="shipment", cascade="all, delete-orphan", order_by="TrackingEvent.event_time"
    )


class TrackingEvent(Base):
    __tablename__ = "tracking_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shipment_id: Mapped[str] = mapped_column(ForeignKey("shipments.id"), index=True)
    event_time: Mapped[str] = mapped_column(String(32))
    location: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(64), default="")
    description: Mapped[str] = mapped_column(Text, default="")

    shipment: Mapped[Shipment] = relationship(back_populates="tracking_events")


class Product(Base):
    __tablename__ = "products"

    sku: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    brand: Mapped[str] = mapped_column(String(64), default="")
    category: Mapped[str] = mapped_column(String(64), default="")
    description: Mapped[str] = mapped_column(Text, default="")


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32), default="")
    body: Mapped[str] = mapped_column(Text)


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[str] = mapped_column(String(32), default="")
    invoice_date: Mapped[str] = mapped_column(String(32), default="")
    amount: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), default="Issued")
    line_summary: Mapped[str] = mapped_column(Text, default="")
    is_latest: Mapped[bool] = mapped_column(default=False)

    customer: Mapped[Customer] = relationship(back_populates="invoices")


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[str] = mapped_column(String(32), default="")
    method: Mapped[str] = mapped_column(String(64), default="")
    amount: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), index=True)
    processed_at: Mapped[str] = mapped_column(String(32), default="")

    customer: Mapped[Customer] = relationship(back_populates="payments")


class License(Base):
    __tablename__ = "licenses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    product_name: Mapped[str] = mapped_column(String(256), default="")
    sku: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), index=True)
    purchased_at: Mapped[str] = mapped_column(String(32), default="")
    expires_at: Mapped[str] = mapped_column(String(32), default="")

    customer: Mapped[Customer] = relationship(back_populates="licenses")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"), index=True)
    plan_name: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(32), default="Active")
    renewal_date: Mapped[str] = mapped_column(String(32), default="")
    renewal_amount: Mapped[str] = mapped_column(String(32), default="")
    downgrade_features_lost: Mapped[str] = mapped_column(Text, default="")

    customer: Mapped[Customer] = relationship(back_populates="subscriptions")


class SupportPlan(Base):
    __tablename__ = "support_plans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    plan_key: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    price_monthly: Mapped[str] = mapped_column(String(32), default="")
    features: Mapped[str] = mapped_column(Text, default="")
    sla_response: Mapped[str] = mapped_column(String(128), default="")


class KnowledgeArticle(Base):
    __tablename__ = "knowledge_articles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text)
