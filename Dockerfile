FROM gfpgan:base
# Thiết lập thư mục làm việc chính
WORKDIR /app/GFPGAN

COPY . .

# Cài đặt GFPGAN ở chế độ phát triển
RUN python setup.py develop

EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "api_gfpgan:app", "--host", "0.0.0.0", "--port", "8000"]
