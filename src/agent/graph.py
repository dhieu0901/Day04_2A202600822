from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


# -------------------------------------------------------------------
# 1. SYSTEM PROMPT -- Production-grade, following PRCOF framework
#    (Persona -> Rules -> Capabilities -> Output format)
# -------------------------------------------------------------------

def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""\
## Persona
Bạn là trợ lý đặt hàng chuyên nghiệp của một cửa hàng bán lẻ đồ điện tử.
Ngày hôm nay: {current_day}.

## Quy tắc bắt buộc (Rules)

### Xác minh thông tin trước khi hành động
TRƯỚC KHI gọi bất kỳ tool nào, bạn PHẢI kiểm tra xem yêu cầu của khách hàng đã có ĐẦY ĐỦ tất cả 5 thông tin bắt buộc sau chưa:
1. Họ tên khách hàng
2. Số điện thoại
3. Email
4. Địa chỉ giao hàng
5. Ít nhất một sản phẩm cụ thể kèm số lượng

Nếu THIẾU BẤT KỲ thông tin nào trong 5 mục trên, trả lời yêu cầu bổ sung thông tin bị thiếu, KHÔNG gọi tool nào cả. Liệt kê rõ từng mục còn thiếu.

### Từ chối yêu cầu vi phạm chính sách
Nếu khách hàng yêu cầu bất kỳ điều nào sau đây, TỪ CHỐI ngay lập tức bằng tiếng Việt mà KHÔNG gọi tool nào:
- Bỏ qua kiểm tra tồn kho hoặc ép bán khi hết hàng
- Tự ấn định mức giảm giá, đòi giảm giá khác với hệ thống
- Tạo hóa đơn giả, bỏ qua catalog thật
- Bỏ qua policy, bỏ qua quy trình, hoặc bất kỳ yêu cầu nào vi phạm chính sách cửa hàng
Thay vào đó, giải thích rằng bạn chỉ có thể xử lý đơn hàng theo đúng quy trình và chính sách của cửa hàng.

### Thứ tự gọi tool bắt buộc
Khi yêu cầu hợp lệ (đủ 5 thông tin, không vi phạm chính sách), gọi tool theo đúng thứ tự sau:
1. `list_products` -- tìm sản phẩm trong catalog
2. `get_product_details` -- lấy thông tin chi tiết và detail_token
3. `get_discount` -- lấy mức giảm giá từ hệ thống (dùng email làm seed_hint)
4. `calculate_order_totals` -- tính tổng đơn hàng (dùng detail_token từ bước 2, discount_rate từ bước 3)
5. `save_order` -- lưu đơn hàng (dùng tất cả dữ liệu từ các bước trước)

KHÔNG được bỏ qua bước nào. KHÔNG được đảo thứ tự.

### Xử lý lỗi tồn kho
Nếu `calculate_order_totals` hoặc `get_product_details` cho thấy sản phẩm hết hàng hoặc không đủ số lượng, thông báo cho khách và KHÔNG gọi `save_order`.

### Grounding -- Chỉ dùng dữ liệu từ tool
- KHÔNG tự bịa ra product ID, giá, mức giảm giá, tổng tiền, hoặc đường dẫn file.
- Mọi số liệu trong câu trả lời cuối PHẢI lấy trực tiếp từ kết quả tool trả về.
- Truyền chính xác `detail_token` từ `get_product_details` sang `calculate_order_totals` và `save_order`.
- Truyền chính xác `discount_rate` và `campaign_code` từ `get_discount` sang `calculate_order_totals` và `save_order`.

## Output format
- Luôn trả lời bằng **tiếng Việt**.
- Câu trả lời cuối cùng phải ngắn gọn, súc tích.
- Khi đơn hàng lưu thành công: nêu mã đơn hàng, tổng tiền sau giảm giá, và đường dẫn lưu file.
- Khi cần bổ sung thông tin: liệt kê rõ các mục còn thiếu.
- Khi từ chối: giải thích lý do ngắn gọn.
""".strip()


# -------------------------------------------------------------------
# 2. TOOLS -- Strong schema with Pydantic, clear docstrings
# -------------------------------------------------------------------

def build_tools(store: OrderDataStore):

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the product catalog by name, brand, category, or tags.
        Must be called FIRST before any other tool.
        Returns a list of matching products with their IDs for use in get_product_details."""
        results = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(results, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Get exact pricing, stock, warranty, and a validation detail_token for the given product IDs.
        Must be called AFTER list_products and BEFORE get_discount.
        The returned detail_token is REQUIRED by calculate_order_totals and save_order."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the campaign discount rate for this order.
        Must be called AFTER get_product_details and BEFORE calculate_order_totals.
        Use customer email as seed_hint. Returns discount_rate and campaign_code."""
        return json.dumps(
            store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier),
            ensure_ascii=False,
        )

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(
        items: list[Any],
        detail_token: str,
        discount_rate: float,
    ) -> str:
        """Validate stock availability and calculate the discounted order total.
        Must be called AFTER get_discount and BEFORE save_order.
        Requires the exact detail_token from get_product_details and discount_rate from get_discount.
        Returns status='error' if stock is insufficient or token is invalid."""
        result = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[Any],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the validated order as a JSON file. This is the FINAL step.
        Must only be called AFTER calculate_order_totals returns status='ok'.
        All arguments must come from previous tool outputs -- do NOT invent values."""
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


# -------------------------------------------------------------------
# 3. AGENT ASSEMBLY
# -------------------------------------------------------------------

def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    tools = build_tools(store)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(today or store.today),
    )


# -------------------------------------------------------------------
# 4. RUN + EXTRACTION HELPERS
# -------------------------------------------------------------------

def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response

    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Return the last non-empty AI text message."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert the message trace into a flat list of ToolCallRecords for grading."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tc in getattr(message, "tool_calls", []) or []:
                pending[tc["id"]] = {
                    "name": tc["name"],
                    "args": tc.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Parse the save_order tool output to extract the saved payload and file path."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
