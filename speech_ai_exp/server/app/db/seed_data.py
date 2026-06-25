from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.orm import Session

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
    TrackingEvent,
)


def _clear_all(session: Session) -> None:
    session.execute(delete(TrackingEvent))
    session.execute(delete(Shipment))
    session.execute(delete(LineItem))
    session.execute(delete(Invoice))
    session.execute(delete(Payment))
    session.execute(delete(License))
    session.execute(delete(Subscription))
    session.execute(delete(Order))
    session.execute(delete(KnowledgeArticle))
    session.execute(delete(SupportPlan))
    session.execute(delete(Customer))
    session.execute(delete(Product))
    session.execute(delete(Policy))
    session.flush()


def seed_demo_data(session: Session) -> None:
    """Load multi-customer demo data aligned with customer-service use cases."""
    _clear_all(session)
    _seed_alex_morgan(session)
    _seed_john_smith(session)
    _seed_sarah_johnson(session)
    _seed_account_c19382(session)
    _seed_nguyen_minh(session)
    _seed_products_and_policies(session)
    _seed_support_plans(session)
    _seed_knowledge_articles(session)
    session.commit()


def _seed_alex_morgan(session: Session) -> None:
    session.add(
        Customer(
            id="CUST-100482",
            preferred_name="Alex",
            full_name="Alexandra Morgan",
            email="alex.morgan@example.test",
            phone="+1-206-555-0148",
            loyalty_tier="Gold",
            loyalty_points=4860,
            store_credit_usd="42.75",
            default_address="4217 Cedar View Lane, Apt 8B, Seattle, WA 98109",
        )
    )
    session.add(
        Order(
            id="ORD-2026-0612-7842",
            display_order_number="7842",
            customer_id="CUST-100482",
            order_date="2026-06-12 10:18",
            order_status="Partially Shipped",
            payment_status="Captured",
            fulfillment_status="Partially Fulfilled",
            order_total="USD 1,486.37",
            is_active=True,
            is_open=True,
        )
    )
    session.add_all(
        [
            LineItem(
                id="LI-7842-01",
                order_id="ORD-2026-0612-7842",
                sku="NB-X15-32-1T",
                product_name="NovaBook X15 Performance Laptop",
                fulfillment_status="Shipped",
                shipment_id="SHP-7842-A",
            ),
            LineItem(
                id="LI-7842-02",
                order_id="ORD-2026-0612-7842",
                sku="MS-ERGO-WL",
                product_name="ArcPoint Ergonomic Wireless Mouse",
                fulfillment_status="Delivered",
                shipment_id="SHP-7842-B",
            ),
        ]
    )
    session.add(
        Shipment(
            id="SHP-7842-A",
            order_id="ORD-2026-0612-7842",
            carrier="UPS",
            tracking_number="1Z999AA10123456784",
            status="In Transit",
            estimated_delivery_date="2026-06-18",
        )
    )


def _seed_john_smith(session: Session) -> None:
    """Use cases 1–3, 8–9, 16–18, 26."""
    session.add(
        Customer(
            id="C-001257",
            preferred_name="John",
            full_name="John Smith",
            email="j.smith@acme.com",
            phone="+1-415-555-0192",
            loyalty_tier="Premium",
            default_address="100 Market Street, San Francisco, CA 94105",
            notes="Acme Corp account. Premium support subscriber.",
        )
    )
    session.add(
        Order(
            id="ORD-784921",
            display_order_number="784921",
            customer_id="C-001257",
            order_date="2026-06-10 14:22",
            order_status="Shipped",
            payment_status="Captured",
            fulfillment_status="Fulfilled",
            order_total="USD 899.00",
            is_active=True,
            is_open=False,
        )
    )
    session.add(
        Order(
            id="ORD-785421",
            display_order_number="785421",
            customer_id="C-001257",
            order_date="2026-06-18 09:15",
            order_status="Processing",
            payment_status="Captured",
            fulfillment_status="Processing",
            order_total="USD 1,299.00",
            is_active=False,
            is_open=True,
            customer_note="Customer corrected order number from 784921 to 785421 in call.",
        )
    )
    session.add(
        LineItem(
            id="LI-784921-01",
            order_id="ORD-784921",
            sku="NB-X15-32-1T",
            product_name="NovaBook X15 Performance Laptop",
            fulfillment_status="Shipped",
            return_window_end="2026-07-10",
            notes="Purchased last week — eligible for return per Gold/Premium window.",
        )
    )
    session.add(
        Order(
            id="ORD-784500",
            display_order_number="784500",
            customer_id="C-001257",
            order_date="2026-06-01 11:00",
            order_status="Delivered",
            payment_status="Captured",
            order_total="USD 4,250.00",
            is_open=False,
        )
    )
    session.add(
        LineItem(
            id="LI-784500-01",
            order_id="ORD-784500",
            sku="SVR-PRO-01",
            product_name="Horizon Pro Server X2000",
            fulfillment_status="Delivered",
            warranty="3-year on-site",
            notes="Delivered 2026-06-05. Within warranty for claim submission.",
        )
    )
    session.add(
        Shipment(
            id="SHP-784921-A",
            order_id="ORD-784921",
            carrier="FedEx",
            tracking_number="7945612345678",
            status="In Transit",
            estimated_delivery_date="2026-06-22",
        )
    )
    session.add(
        Invoice(
            id="INV-2026-784921",
            customer_id="C-001257",
            order_id="ORD-784921",
            invoice_date="2026-06-10",
            amount="USD 899.00",
            status="Issued",
            is_latest=True,
            line_summary="NovaBook X15 Laptop USD 849.00; Shipping USD 25.00; Tax USD 25.00",
        )
    )
    session.add(
        Payment(
            id="PAY-784921-01",
            customer_id="C-001257",
            order_id="ORD-784921",
            method="Visa ending 4421",
            amount="USD 899.00",
            status="Captured",
            processed_at="2026-06-10 14:25",
        )
    )
    session.add_all(
        [
            License(
                id="LIC-001257-01",
                customer_id="C-001257",
                product_name="Horizon Analytics Pro",
                sku="SW-LIC-PRO",
                status="Active",
                purchased_at="2026-05-15",
                expires_at="2027-05-14",
            ),
            License(
                id="LIC-001257-02",
                customer_id="C-001257",
                product_name="Horizon Analytics Pro",
                sku="SW-LIC-PRO",
                status="Active",
                purchased_at="2026-05-15",
                expires_at="2027-05-14",
            ),
            License(
                id="LIC-001257-03",
                customer_id="C-001257",
                product_name="Horizon Analytics Pro",
                sku="SW-LIC-PRO",
                status="Expired",
                purchased_at="2025-05-15",
                expires_at="2026-05-14",
            ),
        ]
    )
    session.add(
        Subscription(
            id="SUB-001257-01",
            customer_id="C-001257",
            plan_name="Premium Support",
            status="Active",
            renewal_date="2026-07-15",
            renewal_amount="USD 49.00",
            downgrade_features_lost="24x7 phone support, 1-hour critical SLA, dedicated TAM",
        )
    )


def _seed_sarah_johnson(session: Session) -> None:
    """Use cases 4, 6, 7, 24."""
    session.add(
        Customer(
            id="C-SJ-2201",
            preferred_name="Sarah",
            full_name="Sarah Johnson",
            email="sarah.johnson@example.test",
            phone="+1-312-555-0177",
            default_address="200 Michigan Ave, Chicago, IL 60601",
        )
    )
    open_orders = [
        ("ORD-918273", "918273", "USD 3,200.00", False, "In Transit", "2026-06-25"),
        ("ORD-918400", "918400", "USD 4,100.00", True, "Delayed", "2026-07-02"),
        ("ORD-918510", "918510", "USD 2,700.00", False, "Processing", ""),
    ]
    for oid, display, total, delayed, status, eta in open_orders:
        session.add(
            Order(
                id=oid,
                display_order_number=display,
                customer_id="C-SJ-2201",
                order_date="2026-06-01 10:00",
                order_status=status,
                payment_status="Captured",
                order_total=total,
                is_active=False,
                is_open=True,
                is_delayed=delayed,
                delay_reason="Carrier weather delay in Memphis hub" if delayed else "",
                cancellation_penalty_usd="USD 150.00" if delayed else "USD 75.00",
            )
        )
    session.add(
        Shipment(
            id="SHP-918273-A",
            order_id="ORD-918273",
            carrier="UPS",
            tracking_number="1Z999AA1091827300",
            status="In Transit",
            estimated_delivery_date="2026-06-25",
        )
    )


def _seed_account_c19382(session: Session) -> None:
    """Use case 5 — customer verification."""
    session.add(
        Customer(
            id="C-19382",
            preferred_name="Chris",
            full_name="Chris Taylor",
            email="chris.taylor@example.test",
            account_status="Active",
            loyalty_tier="Standard",
        )
    )


def _seed_nguyen_minh(session: Session) -> None:
    """Use case 29 — international name and ID."""
    session.add(
        Customer(
            id="C-AX921",
            preferred_name="Minh",
            full_name="Nguyen Thi Minh",
            email="minh.nguyen@example.test",
            phone="+1-408-555-0133",
        )
    )
    session.add(
        Order(
            id="ORD-123456789",
            display_order_number="123456789",
            customer_id="C-AX921",
            order_date="2026-05-20 08:00",
            order_status="Delivered",
            payment_status="Captured",
            order_total="USD 189.99",
            is_open=False,
        )
    )


def _seed_products_and_policies(session: Session) -> None:
    session.add_all(
        [
            Product(
                sku="RT-X500",
                name="Model X500 Enterprise Router",
                brand="Horizon Networking",
                category="Networking",
                description="Wi-Fi 7 (802.11be), tri-band, WPA3-Enterprise, 9.6 Gbps aggregate.",
            ),
            Product(
                sku="SVR-PRO-01",
                name="Horizon Pro Server X2000",
                brand="Horizon",
                category="Servers",
                description="3-year on-site warranty. Rack-mount 2U server.",
            ),
            Product(
                sku="SW-LIC-PRO",
                name="Horizon Analytics Pro License",
                brand="Horizon Software",
                category="Software",
                description="USD 120/seat/year. SSO and audit export included.",
            ),
            Product(
                sku="PRT-M404",
                name="Horizon LaserJet Pro M404",
                brand="Horizon",
                category="Printers",
                description="macOS Ventura certified. Error E57 = fuser temperature fault.",
            ),
        ]
    )
    policies = [
        ("return", "Return Policy", "30-day return window; 45 days for Premium members. Opened unused items eligible with possible restocking fee."),
        ("refund", "Refund Policy", "Refunds within 5 business days after inspection. Opened unused items may qualify if serial matches."),
        ("warranty", "Warranty Policy", "Submit claims with order and serial number at support portal."),
        ("gdpr", "GDPR Retention", "Account data retained 24 months after closure; orders 7 years."),
    ]
    for key, title, body in policies:
        session.add(Policy(policy_key=key, title=title, version="2026-01-01", body=body))


def _seed_support_plans(session: Session) -> None:
    session.add_all(
        [
            SupportPlan(
                id="PLAN-PREMIUM",
                plan_key="premium",
                name="Premium Support",
                price_monthly="USD 49.00",
                features="Email and chat 8x5, 4-hour response, 2 named contacts",
                sla_response="4 business hours",
            ),
            SupportPlan(
                id="PLAN-ENTERPRISE",
                plan_key="enterprise",
                name="Enterprise Support",
                price_monthly="USD 199.00",
                features="24x7 phone/chat/email, dedicated TAM, priority RMA",
                sla_response="1 hour critical",
            ),
        ]
    )


def _seed_knowledge_articles(session: Session) -> None:
    articles = [
        ("kb-x500-wifi7", "product", "Model X500 Wi-Fi 7 Support", "Model X500 supports Wi-Fi 7 802.11be."),
        ("kb-e57", "troubleshooting", "Printer Error E57", "E57 means fuser temperature self-test failed."),
        ("kb-gdpr", "compliance", "GDPR Data Retention", "Account data 24 months after closure; orders 7 years."),
        ("kb-escalation", "workflow", "Human Agent Escalation", "Create ticket ESC-HUMAN and transfer to live queue."),
        ("kb-dr-ha", "compliance", "DR vs HA", "DR: RPO 4h RTO 8h. HA: failover under 60 seconds across AZs."),
    ]
    for aid, cat, title, body in articles:
        session.add(KnowledgeArticle(id=aid, category=cat, title=title, body=body))
