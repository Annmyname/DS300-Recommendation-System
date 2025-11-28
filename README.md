Bản Tiếng Anh phía dưới
# Cải tiến Hệ Thống Khuyến nghị Dựa Trên Nội Dung với Các Phương Pháp Embedding Hiện Đại

## 1. Tóm tắt Dự án

Dự án này tập trung vào việc nghiên cứu và tối ưu hóa hiệu suất của **Hệ thống Khuyến nghị Dựa trên Nội dung (Content-Based Recommender System)** nhằm giải quyết một trong những thách thức lớn nhất là **"bài toán khởi đầu lạnh" (cold start)**.

Chúng tôi triển khai và đánh giá các phương pháp embedding tiên tiến dựa trên Transformer như **Sentence-T5 (ST5)**, **Sentence-BERT (SBERT)**, và **Sentence-GPT (SGPT)**, so sánh với phương pháp cơ bản **TF-IDF**. Các phương pháp này được kết hợp với kỹ thuật giảm chiều **UMAP** và phân cụm **HDBSCAN** hoặc **K-means (k=10)** để tìm ra tổ hợp tối ưu nhất cho hiệu quả gợi ý.

## 2. Kiến trúc Hệ thống (Content-Based Recommendation)

Quy trình xây dựng hệ thống khuyến nghị sách dựa trên nội dung được thực hiện qua các bước chính sau:

1.  **Giai đoạn Embedding:** Đặc trưng văn bản (Tiêu đề, Mô tả, Tác giả) của sách được chuyển đổi thành các vector dày đặc bằng các phương pháp embedding đã thử nghiệm (TF-IDF, ST5, SBERT, SGPT).
2.  **Giảm chiều & Phân cụm:** Áp dụng thuật toán giảm chiều **UMAP** và phân cụm **HDBSCAN** hoặc **K-Means** lên các embedding sách để cải thiện hiệu quả gợi ý.
3.  **Đại diện Người dùng:** Vector đại diện người dùng (user embedding) được tạo ra bằng cách lấy trung bình cộng (average pooling) các embedding của những cuốn sách họ đã đọc.
4.  **Khuyến nghị:** Tính toán **độ tương đồng Cosine** giữa vector người dùng và vector của tất cả các cuốn sách trong cơ sở dữ liệu. Hệ thống đề xuất **Top K** cuốn sách có độ tương đồng cao nhất.

## 3. Bộ dữ liệu

Dự án sử dụng bộ dữ liệu **Goodreads** của Google, tập trung vào danh mục **"truyện tranh và đồ họa" (comic-graphic)**.

Trong thực nghiệm, chúng tôi đã tạo ra 3 bộ dữ liệu khác nhau để đánh giá ảnh hưởng của thông tin đầu vào:
* **Dataset1:** Chỉ sử dụng thuộc tính **Mô tả** (Description).
* **Dataset2:** Kết hợp **Mô tả + Tiêu đề** (Description + Title).
* **Dataset3:** Kết hợp **Mô tả + Tiêu đề + Tác giả** (Description + Title + Author).

## 4. Phương pháp và Công nghệ sử dụng

| Loại Phương pháp | Mô hình/Thuật toán | Vai trò |
| :--- | :--- | :--- |
| **Embedding** | TF-IDF, Sentence-T5 (ST5), Sentence-BERT (SBERT), Sentence-GPT (SGPT) | Tạo vector biểu diễn cho văn bản sách. |
| **Giảm chiều** | UMAP | Giảm chiều dữ liệu vector embedding. |
| **Phân cụm** | HDBSCAN, K-Means | Phân nhóm các cuốn sách tương đồng. |
| **Demo** | Streamlit | Xây dựng ứng dụng web demo đơn giản. |

## 5. Kết quả chính

Kết quả thực nghiệm cho thấy các mô hình sentence-embedding hiện đại vượt trội hơn hẳn so với phương pháp truyền thống TF-IDF.

* **Tổ hợp hiệu suất cao nhất** được ghi nhận là: **SGPT** kết hợp với **UMAP + K-means (k=10)** trên **Dataset3 (Mô tả + Tiêu đề + Tác giả)**.
* **Chỉ số Recall@50 tốt nhất:** **0.4768** (đạt được với tổ hợp SGPT + UMAP + K-means trên Dataset3).

## 6. Demo Ứng dụng

Chúng tôi đã xây dựng một ứng dụng web minh họa đơn giản trên nền tảng **Streamlit**. Người dùng có thể nhập nội dung (từ khóa, tên sách hoặc nội dung cuốn sách) và hệ thống sẽ đề xuất Top K gợi ý, bao gồm Tên sách, Hình bìa sách, Mô tả, và **Độ tương đồng Cosine** với nội dung đầu vào.

## 7. Tác giả

* Nguyễn Lê Vy
* Trần Thị Mỹ Duyên
* Nguyễn Thị Mai Trinh

*(Trường Đại học Công Nghệ Thông Tin, Đại học Quốc gia TP Hồ Chí Minh, Việt Nam)*

***
***

# Improving Content-Based Recommender Systems with Modern Embedding Methods

## 1. Project Summary

This project focuses on researching and optimizing the performance of **Content-Based Recommender Systems** to address the major challenge of the **"cold start" problem**.

We implemented and evaluated advanced Transformer-based embedding methods like **Sentence-T5 (ST5)**, **Sentence-BERT (SBERT)**, and **Sentence-GPT (SGPT)**, comparing them with the baseline **TF-IDF**. These methods are combined with the dimensionality reduction technique **UMAP** and clustering **HDBSCAN** or **K-means (k=10)** to find the optimal combination for recommendation efficiency.

## 2. System Architecture (Content-Based Recommendation)

The process of building the content-based book recommendation system involves the following key steps:

1.  **Embedding Phase:** Book text features (Title, Description, Author) are converted into dense vectors using the tested embedding methods (TF-IDF, ST5, SBERT, SGPT).
2.  **Dimensionality Reduction & Clustering:** The **UMAP** dimensionality reduction algorithm and **HDBSCAN** or **K-Means** clustering are applied to the book embeddings to enhance suggestion quality.
3.  **User Representation:** The user representative vector (user embedding) is created by taking the average pooling of the embeddings of the books they have read.
4.  **Recommendation:** Calculate **Cosine Similarity** between the user vector and the vectors of all books in the database. The system suggests the **Top K** books with the highest similarity.

## 3. Dataset

The project utilizes the Google **Goodreads** dataset, focusing on the **"comic-graphic"** category.

In the experiment, we created 3 different datasets to evaluate the influence of input information:
* **Dataset1:** Using only the **Description** attribute.
* **Dataset2:** Combining **Description + Title**.
* **Dataset3:** Combining **Description + Title + Author**.

## 4. Methods and Technologies Used

| Method Type | Model/Algorithm | Role |
| :--- | :--- | :--- |
| **Embedding** | TF-IDF, Sentence-T5 (ST5), Sentence-BERT (SBERT), Sentence-GPT (SGPT) | Create vector representations for book text. |
| **Dimensionality Reduction** | UMAP | Reduce the dimensionality of the embedding vectors. |
| **Clustering** | HDBSCAN, K-Means | Group similar books. |
| **Demo** | Streamlit | Build a simple web application demo. |

## 5. Key Results

The experimental results demonstrate that modern sentence-embedding models significantly outperform the traditional TF-IDF method.

* **The best performing combination** was recorded as: **SGPT** combined with **UMAP + K-means (k=10)** on **Dataset3 (Description + Title + Author)**.
* **Best Recall@50 score:** **0.4768** (achieved with the SGPT + UMAP + K-means combination on Dataset3).

## 6. Application Demo

We developed a simple illustrative web application using **Streamlit**. Users can input content (keywords, book title, or book content), and the system will suggest the Top K recommendations, including the Book Name, Cover Image, Description, and **Cosine Similarity** with the input content.

## 7. Authors

* Nguyen Le Vy
* Tran Thi My Duyen
* Nguyen Thi Mai Trinh

*(University of Information Technology, Vietnam National University, Ho Chi Minh City, Vietnam)*
