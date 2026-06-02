from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from src.core.schemas import OrderLineInput, ProductRecord


def _normalize(text: str) -> str:
    """Normalize Vietnamese/Unicode text to plain ASCII lowercase for search matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()


class OrderDataStore:
    """
    Manages the product catalog, discount simulation, order pricing, and order persistence.
    All methods return plain dicts so they can be JSON-serialized by the agent tools.
    """

    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or "2026-06-01"

        # Load product catalog
        raw_products = json.loads((self.data_dir / "products.json").read_text(encoding="utf-8"))
        self.products: list[ProductRecord] = [ProductRecord(**item) for item in raw_products]
        self.product_index: dict[str, ProductRecord] = {p.product_id: p for p in self.products}

        # Category aliases for Vietnamese / English synonym support
        self.category_aliases: dict[str, str] = {
            "laptop": "laptop",
            "notebook": "laptop",
            "monitor": "monitor",
            "screen": "monitor",
            "man hinh": "monitor",
            "mouse": "mouse",
            "chuot": "mouse",
            "keyboard": "keyboard",
            "ban phim": "keyboard",
            "headphone": "headphone",
            "tai nghe": "headphone",
            "dock": "dock",
            "storage": "storage",
            "ssd": "storage",
            "stand": "stand",
            "webcam": "webcam",
        }

    # ── Token helpers ──────────────────────────────────────────────

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        """Create a deterministic validation token from a sorted list of product IDs."""
        normalized = "|".join(sorted(product_ids))
        return "DET-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()

    def validate_detail_token(self, product_ids: list[str], detail_token: str) -> bool:
        return detail_token == self.build_detail_token(product_ids)

    # ── Category normalization ─────────────────────────────────────

    def canonicalize_category(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.category_aliases.get(_normalize(value), _normalize(value))

    # ── 1. list_products ───────────────────────────────────────────

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        """Search and filter the product catalog. Returns compact summaries.

        Supports fuzzy matching (typo-tolerant) and keyword overlap scoring
        across product name, brand, tags, and description.
        """
        import difflib

        query_norm = _normalize(query or "")
        q_words = [w for w in query_norm.split() if len(w) > 1 and not w.isdigit()]
        wanted_category = self.canonicalize_category(category)
        wanted_tags = {_normalize(tag) for tag in (required_tags or []) if tag.strip()}
        results: list[tuple[float, int, str, dict]] = []

        for product in self.products:
            if in_stock_only and product.stock <= 0:
                continue
            if wanted_category and product.category != wanted_category:
                continue
            if max_unit_price is not None and product.unit_price > max_unit_price:
                continue

            # Build searchable text blob from all product fields
            haystack = _normalize(
                " ".join([product.name, product.brand, product.category, product.description, *product.tags])
            )
            name_norm = _normalize(product.name)

            score: float = 0.0
            matched_terms: list[str] = []

            # Keyword overlap scoring
            for term in q_words:
                if term in haystack:
                    score += 2.0
                    matched_terms.append(term)

            # Fuzzy match on product name (handles typos like "azus" -> "asus")
            if q_words:
                fuzzy_ratio = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
                score += fuzzy_ratio * 2.5

            # Tag matching bonus
            for tag in wanted_tags:
                if tag in haystack:
                    score += 3.0
                    matched_terms.append(tag)
                else:
                    score -= 1.0

            # Category match bonus
            if wanted_category:
                score += 3.0

            # Skip if query was provided but nothing matched at all
            if q_words and not matched_terms and score < 1.0:
                continue

            results.append(
                (
                    score,
                    product.stock,
                    product.product_id,
                    {
                        "product_id": product.product_id,
                        "name": product.name,
                        "brand": product.brand,
                        "category": product.category,
                        "tags": product.tags,
                        "matched_terms": sorted(set(matched_terms)),
                        "next_step": "Call get_product_details with the chosen product_id list to verify price, stock, and the detail_token.",
                    },
                )
            )

        results.sort(key=lambda item: (-item[0], self.product_index[item[2]].unit_price, item[2]))
        return [item[-1] for item in results[:limit]]

    # ── 2. get_product_details ─────────────────────────────────────

    def get_product_details(self, product_ids: list[str]) -> dict:
        """Return full product info + a validation token for downstream tools."""
        details: list[dict] = []
        for product_id in product_ids:
            product = self.product_index.get(product_id)
            if not product:
                details.append({"product_id": product_id, "status": "not_found"})
                continue
            details.append(
                {
                    "status": "ok",
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "unit_price": product.unit_price,
                    "stock": product.stock,
                    "warranty_months": product.warranty_months,
                    "tags": product.tags,
                    "description": product.description,
                }
            )
        found_ids = [d["product_id"] for d in details if d.get("status") == "ok"]
        return {
            "status": "ok" if found_ids else "error",
            "detail_token": self.build_detail_token(found_ids) if found_ids else "",
            "items": details,
        }

    # ── 3. get_discount ────────────────────────────────────────────

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        """Simulate a campaign discount using deterministic hashing."""
        normalized_seed = seed_hint.strip().lower()
        digest = hashlib.sha256(f"{customer_tier}|{normalized_seed}".encode("utf-8")).hexdigest()
        discount_rate = 0.2 if int(digest[-2:], 16) % 10 < 4 else 0.1
        return {
            "status": "ok",
            "seed_hint": seed_hint,
            "customer_tier": customer_tier,
            "discount_rate": discount_rate,
            "campaign_code": f"FLASH-{int(discount_rate * 100):02d}",
        }

    # ── 4. calculate_order_totals ──────────────────────────────────

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        """Validate stock, verify token, and compute the discounted order total."""
        if discount_rate not in {0.1, 0.2}:
            return {"status": "error", "errors": [f"Unsupported discount rate: {discount_rate}."]}

        requested_ids = [item.product_id for item in items]
        if not self.validate_detail_token(requested_ids, detail_token):
            return {
                "status": "error",
                "errors": ["Invalid detail token. Call get_product_details again before pricing this order."],
            }

        errors: list[str] = []
        lines: list[dict] = []
        subtotal = 0

        for item in sorted(items, key=lambda c: c.product_id):
            product = self.product_index.get(item.product_id)
            if not product:
                errors.append(f"Unknown product_id: {item.product_id}.")
                continue
            if item.quantity > product.stock:
                errors.append(
                    f"Insufficient stock for {product.name}: requested {item.quantity}, available {product.stock}."
                )
                continue
            line_total = product.unit_price * item.quantity
            subtotal += line_total
            lines.append(
                {
                    "product_id": product.product_id,
                    "sku": product.sku,
                    "name": product.name,
                    "category": product.category,
                    "quantity": item.quantity,
                    "unit_price": product.unit_price,
                    "line_total": line_total,
                }
            )

        if errors:
            return {"status": "error", "errors": errors, "items": lines}

        discount_amount = int(subtotal * discount_rate)
        final_total = subtotal - discount_amount
        return {
            "status": "ok",
            "items": lines,
            "pricing": {
                "currency": "VND",
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "final_total": final_total,
            },
            "detail_token": detail_token,
        }

    # ── 5. save_order ──────────────────────────────────────────────

    def save_order(
        self,
        *,
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> dict:
        """Recompute totals, generate a deterministic order ID, and persist the order as JSON."""
        # Re-validate and recompute totals to guarantee correctness
        pricing_snapshot = self.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        if pricing_snapshot["status"] != "ok":
            return pricing_snapshot

        # Deterministic order ID based on customer + items
        normalized_items = sorted(
            [{"product_id": item.product_id, "quantity": item.quantity} for item in items],
            key=lambda c: c["product_id"],
        )
        seed_payload = json.dumps(
            {
                "customer_email": customer_email.strip().lower(),
                "customer_phone": "".join(ch for ch in customer_phone if ch.isdigit()),
                "items": normalized_items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        order_id = "ORD-" + hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:10].upper()
        relative_path = Path("artifacts") / "orders" / f"{order_id}.json"
        absolute_path = self.output_dir / f"{order_id}.json"

        payload = {
            "order_id": order_id,
            "created_at": self.today,
            "status": "confirmed",
            "customer": {
                "name": customer_name.strip(),
                "phone": customer_phone.strip(),
                "email": customer_email.strip(),
                "shipping_address": shipping_address.strip(),
            },
            "items": pricing_snapshot["items"],
            "pricing": pricing_snapshot["pricing"],
            "discount": {
                "campaign_code": campaign_code,
                "customer_tier": customer_tier,
            },
            "notes": notes.strip(),
            "save_path": str(relative_path),
            "source": "llm-order-agent",
        }
        absolute_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "status": "saved",
            "order_id": order_id,
            "path": str(absolute_path),
            "saved_order": payload,
        }
