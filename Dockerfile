FROM gfpgan:base
# Thiết lập thư mục làm việc chính
WORKDIR /app/GFPGAN

COPY api_gfpgan.py /app/GFPGAN

RUN pip install python-dotenv

EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "api_gfpgan:app", "--host", "0.0.0.0", "--port", "8000"]
