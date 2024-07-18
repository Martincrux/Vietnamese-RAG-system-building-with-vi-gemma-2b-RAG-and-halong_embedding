# ==============================================================================
#  This code was written by himmeow the coder.
#  Contact: himmeow.thecoder@gmail.com
#  Discord server: https://discord.gg/deua7trgXc
#
#  Feel free to use and modify this code as you see fit.
# ==============================================================================

"""## Đầu tiên hãy cài đặt các thư viện cần thiết
LƯU Ý: Có thể cần khởi động lại sau khi cài đặt thành công lần đầu *accelerate* 
"""

# pip install -U sentence-transformers
# pip install transformers torch accelerate


"""## Xây dựng embedd model và công cụ truy xuất dữ liệu cho hệ thống"""

from sentence_transformers import SentenceTransformer
import torch

# Khởi tạo model SentenceTransformer (model embedding) với base model là 'hiieu/halong_embedding' từ hugging face
embedd_model = SentenceTransformer("hiieu/halong_embedding")

# --- Giai đoạn 1: Tạo docs_embedding ---

# Dữ liệu thử nghiệm được lấy từ ví dụ xây dựng RAG system với function calling được chia sẻ tại bài viết: https://viblo.asia/p/quy-trinh-xay-dung-he-thong-rag-tich-hop-function-calling-with-source-code-vlZL98GZJQK
docs = [
    "Áo phông nam - thông tin cơ bản: Tên sản phẩm: Áo phông nam Basic; Thương hiệu: The Casual; Nhà sản xuất: Công ty TNHH May mặc Thời trang; Mã sản phẩm: APN001; Loại sản phẩm: Áo thun, Thời trang nam; Mô tả ngắn: Áo phông nam chất liệu cotton thoáng mát, thoải mái cho mọi hoạt động.",
    "Áo phông nam - thông tin chi tiết: - Chất liệu: 100% cotton co giãn 4 chiều, thấm hút mồ hôi tốt. - Thiết kế: Cổ tròn, tay ngắn, form áo suông nhẹ, dễ phối đồ. - Màu sắc: Đen, trắng, xám, xanh navy. - Size: S, M, L, XL, XXL. - Hướng dẫn sử dụng: Giặt máy ở chế độ nhẹ nhàng, không dùng chất tẩy mạnh.",
    "Áo phông nam - thông tin bổ sung: Giá bán: 150.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Giảm 10% cho đơn hàng từ 2 sản phẩm; Đánh giá của khách hàng: 4.5/5 sao - Chất vải mềm mại, thoải mái khi mặc. - Form áo đẹp, dễ phối đồ. - Giá cả hợp lý.",
    "Quần jeans nam - thông tin cơ bản: Tên sản phẩm: Quần jeans nam Slimfit; Thương hiệu: Denim Co.; Nhà sản xuất: Xưởng may jeans Denim; Mã sản phẩm: QJN002; Loại sản phẩm: Quần jeans, Thời trang nam; Mô tả ngắn: Quần jeans nam kiểu dáng slimfit trẻ trung, năng động.",
    "Quần jeans nam - thông tin chi tiết: - Chất liệu: Vải jeans cotton pha spandex co giãn nhẹ, tạo cảm giác thoải mái khi vận động. - Thiết kế: Kiểu dáng slimfit ôm vừa vặn, tôn dáng. - Màu sắc: Xanh đen, xanh nhạt, đen. - Size: 28, 29, 30, 31, 32, 34. - Hướng dẫn sử dụng: Giặt máy với nước lạnh, lộn trái quần khi phơi.",
    "Quần jeans nam - thông tin bổ sung: Giá bán: 350.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Miễn phí vận chuyển cho đơn hàng từ 500.000 VNĐ; Đánh giá của khách hàng: 4.8/5 sao - Chất jeans đẹp, co giãn tốt. - Form quần chuẩn, tôn dáng. - Dịch vụ giao hàng nhanh chóng.",
    "Giày thể thao nam - thông tin cơ bản: Tên sản phẩm: Giày thể thao nam Running; Thương hiệu: SportsPro; Nhà sản xuất: Công ty sản xuất giày dép thể thao ABC; Mã sản phẩm: GTN003; Loại sản phẩm: Giày thể thao, Giày chạy bộ; Mô tả ngắn: Giày thể thao nam thiết kế năng động, phù hợp cho hoạt động chạy bộ và tập luyện thể thao.",
    "Giày thể thao nam - thông tin chi tiết: - Chất liệu: Vải lưới thoáng khí, đế cao su chống trơn trượt. - Công nghệ: Công nghệ đế Ethylene Vinyl Acetate (EVA) êm ái, giảm chấn thương khi vận động. - Màu sắc: Đen, trắng, đỏ, xanh dương. - Size: 39, 40, 41, 42, 43, 44. - Hướng dẫn sử dụng: Vệ sinh giày bằng khăn ẩm, không phơi trực tiếp dưới ánh nắng mặt trời.",
    "Giày thể thao nam - thông tin bổ sung: Giá bán: 500.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Giảm 15% khi mua kèm vớ thể thao; Đánh giá của khách hàng: 4.6/5 sao - Giày nhẹ, êm chân. - Thiết kế đẹp, năng động. - Giá cả phải chăng.",
    "Túi đeo chéo nam - thông tin cơ bản: Tên sản phẩm: Túi đeo chéo nam Canvas; Thương hiệu: Urban Bags; Nhà sản xuất: Xưởng sản xuất túi xách; Mã sản phẩm: TCN004; Loại sản phẩm: Túi đeo chéo, Phụ kiện nam; Mô tả ngắn: Túi đeo chéo nam chất liệu canvas bền đẹp, tiện lợi mang theo khi đi chơi, du lịch.",
    "Túi đeo chéo nam - thông tin chi tiết: - Chất liệu: Vải canvas dày dặn, chống thấm nước nhẹ. - Thiết kế: Ngăn chứa rộng rãi, nhiều ngăn nhỏ tiện lợi, quai đeo chắc chắn, có thể điều chỉnh độ dài. - Màu sắc: Đen, xám, nâu, xanh rêu. - Kích thước: 25cm x 20cm x 8cm. - Hướng dẫn sử dụng: Vệ sinh túi bằng khăn ẩm, không giặt bằng máy giặt.",
    "Túi đeo chéo nam - thông tin bổ sung: Giá bán: 280.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Mua 1 tặng 1 móc khóa thời trang; Đánh giá của khách hàng: 4.7/5 sao - Túi đẹp, chắc chắn. - Nhiều ngăn chứa đồ tiện lợi. - Giá cả hợp lý.",
    "Nón kết nam - thông tin cơ bản: Tên sản phẩm: Nón kết nam Snapback; Thương hiệu: Streetwear; Nhà sản xuất: Công ty sản xuất mũ nón; Mã sản phẩm: NKN005; Loại sản phẩm: Nón kết, Phụ kiện nam; Mô tả ngắn: Nón kết nam kiểu dáng snapback trẻ trung, năng động.",
    "Nón kết nam - thông tin chi tiết: - Chất liệu: Vải kaki dày dặn, form nón cứng cáp. - Thiết kế: Kiểu dáng snapback, có khóa điều chỉnh size phía sau. - Màu sắc: Đen, trắng, xanh, đỏ, vàng. - Chu vi vòng đầu: 56-60cm. - Hướng dẫn sử dụng: Vệ sinh nón bằng khăn ẩm, không giặt bằng máy giặt.",
    "Nón kết nam - thông tin bổ sung: Giá bán: 120.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Giảm 5% cho đơn hàng tiếp theo; Đánh giá của khách hàng: 4.4/5 sao - Nón đẹp, chất lượng tốt. - Giá cả phải chăng.",
    "Ví da nam - thông tin cơ bản: Tên sản phẩm: Ví da nam Bifold; Thương hiệu: Leather Goods; Nhà sản xuất: Xưởng sản xuất đồ da; Mã sản phẩm: VDN006; Loại sản phẩm: Ví da, Phụ kiện nam; Mô tả ngắn: Ví da nam chất liệu da bò thật 100%, thiết kế sang trọng, lịch lãm.",
    "Ví da nam - thông tin chi tiết: - Chất liệu: Da bò thật 100%, đường chỉ may chắc chắn. - Thiết kế: Kiểu dáng ví ngang, nhiều ngăn đựng thẻ, tiền mặt, giấy tờ tùy thân. - Màu sắc: Đen, nâu, nâu đỏ. - Kích thước: 12cm x 9cm x 2cm. - Hướng dẫn sử dụng: Bảo quản nơi khô ráo, tránh tiếp xúc trực tiếp với nước và ánh nắng mặt trời.",
    "Ví da nam - thông tin bổ sung: Giá bán: 550.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Khắc tên miễn phí; Đánh giá của khách hàng: 4.9/5 sao - Chất da đẹp, mềm mại. - Thiết kế sang trọng, tiện dụng. - Dịch vụ khách hàng chu đáo.",
    "Kính mát nam - thông tin cơ bản: Tên sản phẩm: Kính mát nam Polarized; Thương hiệu: Sunnies; Nhà sản xuất: Công ty sản xuất kính mắt; Mã sản phẩm: KMN007; Loại sản phẩm: Kính mát, Phụ kiện nam; Mô tả ngắn: Kính mát nam tròng kính phân cực, chống chói, bảo vệ mắt khỏi tia UV.",
    "Kính mát nam - thông tin chi tiết: - Chất liệu: Gọng kính kim loại cao cấp, tròng kính phân cực Polarized. - Thiết kế: Kiểu dáng thời trang, phù hợp với nhiều khuôn mặt. - Màu sắc: Đen, nâu, xám. - Chức năng: Chống tia UV400, chống chói, bảo vệ mắt. - Hướng dẫn sử dụng: Bảo quản kính trong hộp đựng khi không sử dụng.",
    "Kính mát nam - thông tin bổ sung: Giá bán: 400.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Tặng kèm khăn lau và hộp đựng kính; Đánh giá của khách hàng: 4.7/5 sao - Kính đẹp, chắc chắn. - Tròng kính chống chói hiệu quả. - Giá cả hợp lý.",
    "Tai nghe Bluetooth - thông tin cơ bản: Tên sản phẩm: Tai nghe Bluetooth True Wireless; Thương hiệu: SoundWave; Nhà sản xuất: Công ty sản xuất thiết bị âm thanh; Mã sản phẩm: TNB008; Loại sản phẩm: Tai nghe Bluetooth, Thiết bị âm thanh; Mô tả ngắn: Tai nghe Bluetooth true wireless kết nối ổn định, âm thanh chất lượng cao.",
    "Tai nghe Bluetooth - thông tin chi tiết: - Công nghệ Bluetooth: Bluetooth 5.0, kết nối ổn định trong phạm vi 10m. - Dung lượng pin: 5 giờ nghe nhạc liên tục, hộp sạc cung cấp thêm 20 giờ sử dụng. - Chức năng: Chống ồn, chống nước IPX4, điều khiển cảm ứng. - Màu sắc: Đen, trắng, xanh. - Hướng dẫn sử dụng: Sạc đầy pin trước khi sử dụng lần đầu.",
    "Tai nghe Bluetooth - thông tin bổ sung: Giá bán: 700.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Giảm 10% khi mua online; Đánh giá của khách hàng: 4.8/5 sao - Chất lượng âm thanh tốt. - Kết nối ổn định. - Thời lượng pin lâu.",
    "Sạc dự phòng - thông tin cơ bản: Tên sản phẩm: Sạc dự phòng 10000mAh; Thương hiệu: PowerUp; Nhà sản xuất: Công ty sản xuất pin sạc; Mã sản phẩm: SPD009; Loại sản phẩm: Sạc dự phòng, Phụ kiện điện thoại; Mô tả ngắn: Sạc dự phòng dung lượng 10000mAh, hỗ trợ sạc nhanh cho điện thoại, máy tính bảng.",
    "Sạc dự phòng - thông tin chi tiết: - Dung lượng pin: 10000mAh. - Công suất đầu ra: 5V/2A, 9V/1.5A. - Cổng kết nối: 1 cổng USB-C, 1 cổng USB-A. - Tính năng an toàn: Bảo vệ quá dòng, quá áp, quá nhiệt. - Màu sắc: Đen, trắng, xanh. - Hướng dẫn sử dụng: Sạc đầy pin cho sạc dự phòng trước khi sử dụng.",
    "Sạc dự phòng - thông tin bổ sung: Giá bán: 250.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Tặng kèm cáp sạc micro USB; Đánh giá của khách hàng: 4.6/5 sao - Dung lượng pin lớn. - Sạc nhanh chóng. - Thiết kế nhỏ gọn, tiện lợi.",
    "Balo laptop - thông tin cơ bản: Tên sản phẩm: Balo laptop 15.6 inch; Thương hiệu: CityGear; Nhà sản xuất: Xưởng sản xuất balo túi xách; Mã sản phẩm: BLT010; Loại sản phẩm: Balo laptop, Phụ kiện công nghệ; Mô tả ngắn: Balo laptop thiết kế hiện đại, nhiều ngăn chứa, bảo vệ laptop an toàn.",
    "Balo laptop - thông tin chi tiết: - Chất liệu: Vải polyester chống thấm nước, chống xước. - Kích thước: 45cm x 30cm x 15cm. - Ngăn chứa: Ngăn laptop 15.6 inch, ngăn đựng ipad, nhiều ngăn phụ kiện. - Quai đeo: Êm ái, chắc chắn, có thể điều chỉnh độ dài. - Màu sắc: Đen, xám, xanh navy. - Hướng dẫn sử dụng: Không giặt bằng máy giặt, vệ sinh bằng khăn ẩm.",
    "Balo laptop - thông tin bổ sung: Giá bán: 380.000 VNĐ; Tình trạng tồn kho: Còn hàng; Khuyến mãi: Freeship cho đơn hàng từ 300.000 VNĐ; Đánh giá của khách hàng: 4.7/5 sao - Balo đẹp, chắc chắn. - Nhiều ngăn chứa đồ tiện lợi. - Giá cả phải chăng."
]
# Tạo embeddings cho các tài liệu và lưu vào biến docs_embeddings
docs_embeddings = embedd_model.encode(docs)


# --- Giai đoạn 2: Hàm truy xuất tài liệu ---

def retrieve_relevant_docs(query: str, top_k: int = 3) -> str:
    """
    Truy xuất các tài liệu liên quan nhất đến câu truy vấn.

    Args:
        query: Câu truy vấn.
        top_k: Số lượng tài liệu muốn truy xuất.

    Returns:
        Chuỗi chứa các tài liệu liên quan, cách nhau bằng dấu xuống dòng.
    """
    # Mã hóa câu truy vấn thành vector embedding
    query_embedding = embedd_model.encode([query])
    # Tính toán cosine similarity giữa câu truy vấn và các tài liệu
    similarities = embedd_model.similarity(query_embedding, docs_embeddings).flatten()
    # Lấy ra top k tài liệu có cosine similarity cao nhất
    _, sorted_indices = torch.topk(similarities, top_k)
    # Trả về chuỗi chứa các tài liệu liên quan
    return "\n\n".join([docs[idx] for idx in sorted_indices])

# Thử nghiệm với task sử dụng hàm retrieve_relevant_docs để truy xuất 5 tài liệu liên quan nhất với query dưới đây
query = "thông tin về áo phông nam?"
relevant_docs = retrieve_relevant_docs(query, top_k=5)
print(relevant_docs)


"""## Kết hợp gen model với hệ thống truy xuất dữ liệu để tạo RAG system hoàn chỉnh"""

from transformers import AutoTokenizer, AutoModelForCausalLM

# Khởi tạo tokenizer và model RAG
tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
model = AutoModelForCausalLM.from_pretrained(
    "himmeow/vi-gemma-2b-RAG",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Sử dụng GPU nếu có
if torch.cuda.is_available():
    model.to("cuda")

# Định dạng prompt cho model
prompt = """
### Instruction and Input:

Dựa vào ngữ cảnh/tài liệu sau:
{}

Hãy trả lời câu hỏi: {}

### Response:
{}
"""

# Hàm thực hiện quy trình RAG
def generate_answer(query: str) -> str:
    """
    Thực hiện quy trình Retrieval Augmented Generation (RAG) để trả lời câu hỏi.

    Args:
        query: Câu hỏi cần trả lời.

    Returns:
        Câu trả lời được tạo bởi model RAG.
    """
    # Truy xuất tài liệu liên quan
    relevant_docs = retrieve_relevant_docs(query, top_k=3)
    # Định dạng input text
    input_text = prompt.format(relevant_docs, query, " ")
    # Mã hóa input text thành input ids
    input_ids = tokenizer(input_text, return_tensors="pt")
    # Sử dụng GPU cho input ids nếu có
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    # Tạo văn bản bằng model
    outputs = model.generate(
        **input_ids,
        max_new_tokens=500,
        no_repeat_ngram_size=7,  # Ngăn chặn lặp lại các cụm từ 7 gram
        do_sample=True,   # Kích hoạt chế độ tạo văn bản dựa trên lấy mẫu. Trong chế độ này, model sẽ chọn ngẫu nhiên token tiếp theo dựa trên xác suất được tính từ phân phối xác suất của các token.
        temperature=0.2,  # Giảm temperature để kiểm soát tính ngẫu nhiên
        # early_stopping=True,  # Dừng tạo văn bản khi tìm thấy kết thúc phù hợp
        )
    # Giải mã và trả về kết quả
    return tokenizer.decode(outputs[0])

# Sử dụng hàm generate_answer để trả lời câu hỏi
query = "Cho tôi thông tin về áo phông nam ở cửa hàng, bao gồm giá bán, chất liệu và khuyến mãi"
answer = generate_answer(query)
print(answer)