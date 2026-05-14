📌 Baseline Model Evaluation (Naive Forecast)

Trong nghiên cứu này, mô hình Naive Forecast được sử dụng như một mô hình cơ sở (baseline) để đánh giá hiệu suất dự đoán giá thanh long. Dữ liệu được chia theo tỷ lệ 70% huấn luyện và 30% kiểm tra, đảm bảo đúng nguyên tắc của bài toán chuỗi thời gian (time series) và tránh hiện tượng rò rỉ dữ liệu (data leakage).

Kết quả đánh giá trên tập kiểm tra như sau:

Ruột Trắng:
MAE = 3338.56
RMSE = 3740.48



Ruột Đỏ:
MAE = 5149.75
RMSE = 5886.67



Mô hình Naive dự đoán giá trị tại thời điểm t+1 bằng giá trị tại thời điểm t cuối cùng của tập huấn luyện. Do đó, mô hình này không học tham số mà được sử dụng làm mốc so sánh (benchmark) cho các mô hình dự báo nâng cao trong các bước tiếp theo.

Các mô hình phát triển sau này cần đạt sai số thấp hơn mô hình Naive để chứng minh khả năng cải thiện hiệu suất dự báo.