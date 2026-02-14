import requests
import fitz  # PyMuPDF

def create_dummy_pdf(filename="test.pdf"):
    doc = fitz.open()
    page = doc.new_page()
    text = """
    Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.
    
    There are two types of photosynthetic processes: oxygenic photosynthesis and anoxygenic photosynthesis.
    The general equation for photosynthesis is:
    6CO2 + 6H2O + Light Energy -> C6H12O6 + 6O2
    
    This process takes place in the chloroplasts, specifically using chlorophyll.
    """
    page.insert_text((50, 50), text)
    doc.save(filename)
    print(f"Created {filename}")

def test_health():
    try:
        r = requests.get("http://localhost:8000/")
        print("Health Check:", r.status_code, r.json())
    except Exception as e:
        print("Health Check Failed:", e)

def test_upload():
    create_dummy_pdf()
    files = {'file': open('test.pdf', 'rb')}
    try:
        print("Sending request...")
        r = requests.post("http://localhost:8000/api/v1/process", files=files, timeout=60)
        print("Status:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Success!")
            print("Meta:", data.get("meta"))
            print("Exam Questions:", len(data["data"]["exam"]["mcq"]))
        else:
            print("Error:", r.text)
    except Exception as e:
        print("Upload Failed:", e)

if __name__ == "__main__":
    test_health()
    # test_upload() # Uncomment to test full flow if needed, but health check is first priority
