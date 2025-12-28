import time
from fastapi import FastAPI, UploadFile, File
import oci
import base64
from db_util import init_db, save_inv_extraction


app = FastAPI()

# Load OCI config from ~/.oci/config
config = oci.config.from_file()

doc_client = oci.ai_document.AIServiceDocumentClient(config)



@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    # Base64 encode PDF
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    

    document = oci.ai_document.models.InlineDocumentDetails(
        data=encoded_pdf
    )
    
    request = oci.ai_document.models.AnalyzeDocumentDetails(
        document=document,
        features=[
            oci.ai_document.models.DocumentFeature(
                feature_type="KEY_VALUE_EXTRACTION"
            ),
            oci.ai_document.models.DocumentClassificationFeature(
                max_results=5
            )
        ]
    )

    response = doc_client.analyze_document(request)

    data = {}
    data_confidence = {}

    for page in response.data.pages:
        if page.document_fields:
            for field in page.document_fields:
                field_name = field.field_label.name
                field_confidence = field.field_label.confidence if field.field_label.confidence else None
                field_value = field.field_value.text
            
                data[field_name] = field_value
                data_confidence[field_name] = field_confidence



    result = {
        "confidence": "1",
        "data": data,
        "dataConfidence": data_confidence
    }

    save_inv_extraction(result)

    return result



@app.get('/health')
def health():
    return {'status': 'ok'}




if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8080)