from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
model = AutoModelForCausalLM.from_pretrained(
    "himmeow/vi-gemma-2b-RAG",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Move the model to the GPU if available
if torch.cuda.is_available():
    model.to("cuda") 

prompt = """
### Instruction and Input:
Dựa vào ngữ cảnh/tài liệu sau: 
{}
Hãy trả lời câu hỏi: {}

### Response:
{}
"""

input_data = """
Short Tandem Repeats (STRs) là các trình tự DNA lặp lại ngắn (2- 6 nucleotides) xuất hiện phổ biến trong hệ gen của con người. Các trình tự này có tính đa hình rất cao trong tự nhiên, điều này khiến các STRs trở thành những markers di truyền rất quan trọng trong nghiên cứu bản đồ gen người và chuẩn đoán bệnh lý di truyền cũng như xác định danh tính trong lĩnh vực pháp y.
Các STRs trở nên phổ biến tại các phòng xét nghiệm pháp y bởi vì việc nhân bản và phân tích STRs chỉ cần lượng DNA rất thấp ngay cả khi ở dạng bị phân hủy việc đinh danh vẫn có thể được thực hiện thành công. Hơn nữa việc phát hiện và đánh giá sự nhiễm DNA mẫu trong các mẫu vật có thể được giải quyết nhanh với kết quả phân tích STRs. Ở Hoa Kỳ hiện nay, từ bộ 13 markers nay đã tăng lên 20 markers chính đang được sử dụng để tạo ra một cơ sở dữ liệu DNA trên toàn đất nước được gọi là The FBI Combined DNA Index System (Expaned CODIS).
CODIS và các cơ sử dữ liệu DNA tương tự đang được sử dụng thực sự thành công trong việc liên kết các hồ sơ DNA từ các tội phạm và các bằng chứng hiện trường vụ án. Kết quả định danh STRs cũng được sử dụng để hỗ trợ hàng trăm nghìn trường hợp xét nghiệm huyết thống cha con mỗi năm'
"""

query = "Hãy cho tôi biết một số tính chất của STRs được dùng để làm gì?"

input_text = prompt.format(input_data, query," ",)

input_ids = tokenizer(input_text, return_tensors="pt")

if torch.cuda.is_available():
    input_ids = input_ids.to("cuda") # Move input_ids to the same device as the model


outputs = model.generate(
    **input_ids,
    max_new_tokens=500,
    temperature=0.7,  # Giảm temperature để kiểm soát tính ngẫu nhiên 
    no_repeat_ngram_size=5,  # Ngăn chặn lặp lại các cụm từ 5 gram
    # early_stopping=True,  # Dừng tạo văn bản khi tìm thấy kết thúc phù hợp
)
print(tokenizer.decode(outputs[0]))