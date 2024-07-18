# ==============================================================================
#  This code was written by hiieu && himmeow the coder.
#  Contact: https://www.linkedin.com/in/hieu-ngo-500818174
#           himmeow.thecoder@gmail.com
#  Discord server: https://discord.gg/deua7trgXc
#
#  Feel free to use and modify this code as you see fit.
# ==============================================================================

"""## Đầu tiên hãy cài đặt các thư viện cần thiết"""

"""## Cài đặt và sử dụng model"""

from sentence_transformers import SentenceTransformer
import torch

# Tải model từ Hugging Face Hub
model = SentenceTransformer("hiieu/halong_embedding")

# Câu truy vấn và danh sách tài liệu
query = "Bóng đá có lợi ích gì cho sức khỏe?"
docs = [
    "Bóng đá giúp cải thiện sức khỏe tim mạch và tăng cường sức bền.",
    "Bóng đá là môn thể thao phổ biến nhất thế giới.",
    "Chơi bóng đá giúp giảm căng thẳng và cải thiện tâm lý.",
    "Bóng đá có thể giúp bạn kết nối với nhiều người hơn.",
    "Bóng đá không chỉ là môn thể thao mà còn là cách để giải trí."
]

# Mã hóa câu truy vấn và tài liệu thành vector embedding
query_embedding = model.encode([query])
doc_embeddings = model.encode(docs)

# Tính toán cosine similarity giữa câu truy vấn và các tài liệu
similarities = model.similarity(query_embedding, doc_embeddings).flatten()

# Sắp xếp tài liệu theo cosine similarity giảm dần
sorted_indices = torch.argsort(similarities, descending=True)
sorted_docs = [docs[idx] for idx in sorted_indices]
sorted_scores = [similarities[idx].item() for idx in sorted_indices]

# In ra các tài liệu đã sắp xếp cùng với điểm số cosine similarity
for doc, score in zip(sorted_docs, sorted_scores):
    print(f"Tài liệu: {doc} - Điểm số Cosine Similarity: {score:.4f}")