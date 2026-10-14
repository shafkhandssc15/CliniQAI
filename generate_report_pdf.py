import os
import base64
from playwright.sync_api import sync_playwright
import pypdf

def main():
    print("Reading report_template.html...")
    template_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    logo_path = os.path.join(os.path.dirname(__file__), "Logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'<img src="data:image/png;base64,{b64}" class="brand-logo" alt="CliniQ AI">'
    else:
        logo_html = '<div style="font-size:24pt;font-weight:800;color:#0284c7;">CliniQ AI</div>'

    html = html.replace("{{LOGO_ELEMENT}}", logo_html)

    temp_html_path = os.path.join(os.path.dirname(__file__), "temp_report.html")
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    pdf_output_path = os.path.join(os.path.dirname(__file__), "CliniQ_AI_Complete_Project_Report.pdf")

    print("Launching Chromium via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page_url = f"file:///{temp_html_path.replace(os.sep, '/')}"
        page.goto(page_url)
        page.wait_for_load_state("networkidle")
        
        print("Rendering PDF...")
        page.pdf(
            path=pdf_output_path,
            format="A4",
            print_background=True,
            margin={
                "top": "16mm",
                "bottom": "18mm",
                "left": "16mm",
                "right": "16mm"
            },
            display_header_footer=True,
            header_template='<div style="font-size:8px; font-family:\'Segoe UI\',sans-serif; color:#64748b; width:100%; text-align:right; padding-right:16mm; font-weight:600;">CliniQ AI — Full Project Technical Analysis & Clinical Architecture Audit</div>',
            footer_template='<div style="font-size:8px; font-family:\'Segoe UI\',sans-serif; color:#64748b; width:100%; text-align:center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span> • Confidential Medical Informatics Report • SLMC Standard</div>'
        )
        browser.close()

    if os.path.exists(temp_html_path):
        os.remove(temp_html_path)

    pdf_size = os.path.getsize(pdf_output_path)
    reader = pypdf.PdfReader(pdf_output_path)
    num_pages = len(reader.pages)

    print("=" * 60)
    print("SUCCESS: PDF REPORT CREATED")
    print(f"Path: {pdf_output_path}")
    print(f"File Size: {pdf_size:,} bytes")
    print(f"Page Count: {num_pages} pages")
    print("=" * 60)

if __name__ == "__main__":
    main()
