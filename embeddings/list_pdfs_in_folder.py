import os

def list_pdfs_in_folder():
    # ✅ Ask for folder path
    pdf_folder = input("📂 Enter the folder containing PDFs: ").strip().strip('"')

    # ✅ Check if folder exists
    if not os.path.exists(pdf_folder) or not os.path.isdir(pdf_folder):
        print("❌ Folder not found! Please check the path.")
        return

    # ✅ List all PDF files in the folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in this folder.")
    else:
        print(f"✅ Found {len(pdf_files)} PDFs:")
        for pdf in pdf_files:
            print("  •", pdf)

if __name__ == "__main__":
    list_pdfs_in_folder()
