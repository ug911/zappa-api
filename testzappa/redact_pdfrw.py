import pdfrw

def redact_phone_numbers(input_pdf: str, output_pdf: str):
    # Load the input PDF
    pdf = pdfrw.PdfReader(input_pdf)

    # Define a function to redact text
    def redact_text(page, text_to_redact):
        for annot in page.Annots:
            if annot.Subtype == "/Text":
                if text_to_redact in annot.get("/Contents", ""):
                    annot.update(pdfrw.PdfDict(Subtype="/Redact"))

    # Specify the text (phone numbers) to redact
    phone_numbers_to_redact = ["123-456-7890", "987-654-3210"]

    # Redact phone numbers on each page
    for page in pdf.pages:
        for phone_number in phone_numbers_to_redact:
            redact_text(page, phone_number)

    # Save the redacted PDF
    pdfrw.PdfWriter().write(output_pdf, pdf)

if __name__ == "__main__":
    input_pdf_path = "your_input.pdf"
    output_pdf_path = "redacted_output.pdf"
    redact_phone_numbers(input_pdf_path, output_pdf_path)
