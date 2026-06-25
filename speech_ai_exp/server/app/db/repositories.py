from __future__ import annotations

import re

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    Customer,
    Invoice,
    KnowledgeArticle,
    License,
    LineItem,
    Order,
    Payment,
    Policy,
    Product,
    Shipment,
    Subscription,
    SupportPlan,
)
from app.db.normalize import normalize_customer_id, normalize_email, order_id_candidates


class CommerceRepository:
    """Structured lookups for customer profiles, orders, billing, and licenses."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def lookup_customer(
        self,
        *,
        customer_id: str = "",
        email: str = "",
        name: str = "",
    ) -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email, name=name)
            if customer is None:
                return "No matching customer found."
            return self._format_customer(session, customer)

    def lookup_order(self, *, order_id: str = "") -> str:
        with self._session_factory() as session:
            order = self._find_order(session, order_id)
            if order is None:
                return f"No order found for {order_id!r}."
            return self._format_order(order)

    def lookup_invoice(
        self,
        *,
        customer_id: str = "",
        email: str = "",
        invoice_id: str = "",
        latest: bool = False,
    ) -> str:
        with self._session_factory() as session:
            if invoice_id.strip():
                inv = session.get(Invoice, invoice_id.strip())
                if inv is None:
                    return f"No invoice {invoice_id!r}."
                return self._format_invoice(inv)
            customer = self._find_customer(session, customer_id=customer_id, email=email)
            if customer is None:
                return "No matching customer found."
            q = select(Invoice).where(Invoice.customer_id == customer.id)
            if latest:
                q = q.where(Invoice.is_latest.is_(True))
            inv = session.scalar(q.order_by(Invoice.invoice_date.desc()))
            if inv is None:
                return f"No invoices for customer {customer.id}."
            return self._format_invoice(inv)

    def lookup_payment_status(self, *, email: str = "", customer_id: str = "", order_id: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email)
            if customer is None and not order_id.strip():
                return "No matching customer found."
            q = select(Payment)
            if customer is not None:
                q = q.where(Payment.customer_id == customer.id)
            if order_id.strip():
                order = self._find_order(session, order_id)
                if order is None:
                    return f"No order found for {order_id!r}."
                q = q.where(Payment.order_id == order.id)
            payments = session.scalars(q.order_by(Payment.processed_at.desc()).limit(5)).all()
            if not payments:
                return "No payment records found."
            lines = ["Payment records:"]
            for p in payments:
                lines.append(
                    f"- {p.id}: order {p.order_id}, {p.amount}, status {p.status}, "
                    f"method {p.method}, processed {p.processed_at}"
                )
            return "\n".join(lines)

    def list_open_orders(self, *, customer_id: str = "", email: str = "", name: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email, name=name)
            if customer is None:
                return "No matching customer found."
            orders = session.scalars(
                select(Order)
                .where(Order.customer_id == customer.id, Order.is_open.is_(True))
                .order_by(Order.order_date.desc())
            ).all()
            if not orders:
                return f"No open orders for {customer.preferred_name} ({customer.id})."
            lines = [f"Open orders for {customer.full_name} ({customer.id}):"]
            for o in orders:
                delay = " [DELAYED]" if o.is_delayed else ""
                lines.append(
                    f"- {o.display_order_number or o.id}{delay}: {o.order_status}, "
                    f"total {o.order_total}, cancel penalty {o.cancellation_penalty_usd}"
                )
            return "\n".join(lines)

    def list_delayed_orders(self, *, customer_id: str = "", email: str = "", name: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email, name=name)
            if customer is None:
                return "No matching customer found."
            orders = session.scalars(
                select(Order)
                .where(Order.customer_id == customer.id, Order.is_delayed.is_(True))
                .order_by(Order.order_date.desc())
            ).all()
            if not orders:
                return f"No delayed orders for {customer.preferred_name}."
            lines = [f"Delayed orders for {customer.full_name}:"]
            for o in orders:
                lines.append(
                    f"- Order {o.display_order_number or o.id}: {o.order_status}. "
                    f"Reason: {o.delay_reason or 'unspecified'}"
                )
            return "\n".join(lines)

    def lookup_licenses(self, *, customer_id: str = "", email: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email)
            if customer is None:
                return "No matching customer found."
            licenses = session.scalars(
                select(License).where(License.customer_id == customer.id)
            ).all()
            if not licenses:
                return f"No licenses for {customer.id}."
            active = sum(1 for lic in licenses if lic.status.lower() == "active")
            lines = [
                f"Licenses for {customer.full_name} ({customer.id}): "
                f"{active} active of {len(licenses)} total"
            ]
            for lic in licenses:
                lines.append(f"- {lic.product_name} ({lic.id}): {lic.status}, expires {lic.expires_at}")
            return "\n".join(lines)

    def lookup_subscription(self, *, customer_id: str = "", email: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email)
            if customer is None:
                return "No matching customer found."
            sub = session.scalar(
                select(Subscription)
                .where(Subscription.customer_id == customer.id)
                .order_by(Subscription.renewal_date.desc())
            )
            if sub is None:
                return f"No subscription for {customer.id}."
            return (
                f"Subscription {sub.id}\n"
                f"Plan: {sub.plan_name}\n"
                f"Status: {sub.status}\n"
                f"Renewal date: {sub.renewal_date}\n"
                f"Renewal amount: {sub.renewal_amount}\n"
                f"If downgraded, features lost: {sub.downgrade_features_lost}"
            )

    def compare_support_plans(self) -> str:
        with self._session_factory() as session:
            plans = session.scalars(select(SupportPlan).order_by(SupportPlan.plan_key)).all()
            if not plans:
                return "No support plans configured."
            parts = ["Support plan comparison:"]
            for p in plans:
                parts.append(
                    f"\n{p.name} ({p.price_monthly}/month)\n"
                    f"SLA: {p.sla_response}\n"
                    f"Features: {p.features}"
                )
            return "\n".join(parts)

    def list_customer_orders(self, *, customer_id: str = "", email: str = "", name: str = "", limit: int = 5) -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email, name=name)
            if customer is None:
                return "No matching customer found."
            orders = session.scalars(
                select(Order)
                .where(Order.customer_id == customer.id)
                .order_by(Order.is_active.desc(), Order.order_date.desc())
                .limit(max(1, min(limit, 10)))
            ).all()
            if not orders:
                return f"No orders for customer {customer.id}."
            lines = [f"Orders for {customer.preferred_name} ({customer.id}):"]
            for o in orders:
                flags = []
                if o.is_active:
                    flags.append("ACTIVE")
                if o.is_open:
                    flags.append("OPEN")
                if o.is_delayed:
                    flags.append("DELAYED")
                flag_s = f" [{', '.join(flags)}]" if flags else ""
                lines.append(
                    f"- {o.display_order_number or o.id}{flag_s}: {o.order_status}, "
                    f"total {o.order_total}, date {o.order_date}"
                )
            return "\n".join(lines)

    def get_active_order(self, *, customer_id: str = "", email: str = "", name: str = "") -> str:
        with self._session_factory() as session:
            customer = self._find_customer(session, customer_id=customer_id, email=email, name=name)
            if customer is None:
                return "No matching customer found."
            order = session.scalar(
                select(Order)
                .options(
                    joinedload(Order.line_items),
                    joinedload(Order.shipments).joinedload(Shipment.tracking_events),
                )
                .where(Order.customer_id == customer.id, Order.is_active.is_(True))
                .order_by(Order.order_date.desc())
            )
            if order is None:
                open_orders = session.scalars(
                    select(Order).where(Order.customer_id == customer.id, Order.is_open.is_(True))
                ).all()
                if open_orders:
                    return self.list_open_orders(customer_id=customer.id)
                return f"No active order for customer {customer.id}."
            return self._format_order(order)

    def iter_vector_documents(self) -> list[tuple[str, str, str]]:
        docs: list[tuple[str, str, str]] = []
        with self._session_factory() as session:
            for customer in session.scalars(select(Customer)).all():
                docs.append((f"customer:{customer.id}", "database/customers", self._format_customer(session, customer)))
            for order in session.scalars(
                select(Order).options(
                    joinedload(Order.line_items),
                    joinedload(Order.shipments).joinedload(Shipment.tracking_events),
                )
            ).unique().all():
                docs.append((f"order:{order.id}", "database/orders", self._format_order(order)))
            for product in session.scalars(select(Product)).all():
                text = f"SKU: {product.sku}\nProduct: {product.name}\nBrand: {product.brand}\n{product.description}"
                docs.append((f"product:{product.sku}", "database/products", text))
            for policy in session.scalars(select(Policy)).all():
                docs.append((f"policy:{policy.policy_key}", "database/policies", f"{policy.title}\n{policy.body}"))
            for inv in session.scalars(select(Invoice)).all():
                docs.append((f"invoice:{inv.id}", "database/invoices", self._format_invoice(inv)))
            for lic in session.scalars(select(License)).all():
                docs.append(
                    (f"license:{lic.id}", "database/licenses", f"{lic.product_name} status {lic.status} expires {lic.expires_at}")
                )
            for art in session.scalars(select(KnowledgeArticle)).all():
                docs.append((f"kb:{art.id}", f"database/kb/{art.category}", f"{art.title}\n{art.body}"))
            for plan in session.scalars(select(SupportPlan)).all():
                docs.append(
                    (f"plan:{plan.plan_key}", "database/support_plans", f"{plan.name}\n{plan.features}\nSLA: {plan.sla_response}")
                )
        return docs

    def _find_order(self, session: Session, order_id: str) -> Order | None:
        for candidate in order_id_candidates(order_id):
            order = session.scalar(
                select(Order)
                .options(
                    joinedload(Order.line_items),
                    joinedload(Order.shipments).joinedload(Shipment.tracking_events),
                    joinedload(Order.customer),
                )
                .where(or_(Order.id == candidate, Order.display_order_number == candidate))
            )
            if order is not None:
                return order
        return None

    def _find_customer(
        self,
        session: Session,
        *,
        customer_id: str = "",
        email: str = "",
        name: str = "",
    ) -> Customer | None:
        if customer_id.strip():
            norm = normalize_customer_id(customer_id)
            raw_id = customer_id.strip().upper()
            if not raw_id.startswith("C") and re.fullmatch(r"[A-Z0-9-]+", raw_id):
                raw_id = f"C-{raw_id}"
            candidates = {
                customer_id.strip(),
                customer_id.strip().upper(),
                raw_id,
                norm,
                norm.replace("-", ""),
            }
            for cid in candidates:
                hit = session.get(Customer, cid)
                if hit is not None:
                    return hit
        if email.strip():
            norm_email = normalize_email(email)
            hit = session.scalar(select(Customer).where(Customer.email == norm_email))
            if hit is not None:
                return hit
        if name.strip():
            needle = name.strip().lower()
            for customer in session.scalars(select(Customer)).all():
                if needle in customer.preferred_name.lower() or needle in customer.full_name.lower():
                    return customer
        return None

    def _format_customer(self, session: Session, customer: Customer) -> str:
        open_count = session.scalar(
            select(func.count()).select_from(Order).where(
                Order.customer_id == customer.id, Order.is_open.is_(True)
            )
        )
        return (
            f"Customer ID: {customer.id}\n"
            f"Name: {customer.full_name} ({customer.preferred_name})\n"
            f"Email: {customer.email}\n"
            f"Phone: {customer.phone}\n"
            f"Tier: {customer.loyalty_tier}\n"
            f"Open orders: {open_count or 0}\n"
            f"Address: {customer.default_address}"
        )

    def _format_invoice(self, inv: Invoice) -> str:
        return (
            f"Invoice ID: {inv.id}\n"
            f"Customer ID: {inv.customer_id}\n"
            f"Order ID: {inv.order_id}\n"
            f"Date: {inv.invoice_date}\n"
            f"Amount: {inv.amount}\n"
            f"Status: {inv.status}\n"
            f"Line items: {inv.line_summary}"
        )

    def _format_order(self, order: Order) -> str:
        lines = [
            f"Order ID: {order.id}",
            f"Order number: {order.display_order_number or order.id}",
            f"Customer ID: {order.customer_id}",
            f"Order date: {order.order_date}",
            f"Status: {order.order_status}",
            f"Payment: {order.payment_status}",
            f"Total: {order.order_total}",
        ]
        if order.is_delayed:
            lines.append(f"DELAYED: {order.delay_reason}")
        if order.cancellation_penalty_usd and order.cancellation_penalty_usd != "0.00":
            lines.append(f"Cancellation penalty: {order.cancellation_penalty_usd}")
        for item in order.line_items:
            lines.append(f"- {item.product_name} (SKU {item.sku}): {item.fulfillment_status}")
        for ship in order.shipments:
            lines.append(
                f"Shipment {ship.id}: {ship.carrier} {ship.tracking_number}, "
                f"status {ship.status}, ETA {ship.estimated_delivery_date or ship.delivery_date}"
            )
            for ev in ship.tracking_events:
                lines.append(f"  · {ev.event_time} {ev.location}: {ev.status}")
        return "\n".join(lines)
