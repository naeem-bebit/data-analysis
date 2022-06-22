from PyPDF2 import PdfWriter, PdfReader
from pdfminer.layout import LTTextBoxHorizontal
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
import io
import re
from pdfminer3.converter import TextConverter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfpage import PDFPage
from pdfminer3.layout import LAParams, LTTextBox
from PyPDF2 import PdfFileReader
from PyPDF2 import PdfFileReader, PdfFileWriter

file_path = 'sample2.pdf'
pdf = PdfFileReader(file_path)

with open('sample2.txt', 'w') as f:
    for page_num in range(pdf.numPages):
        # print('Page: {0}'.format(page_num))
        pageObj = pdf.getPage(page_num)

        try:
            txt = pageObj.extractText()
            print(''.center(100, '-'))
            print(txt)
        except:
            pass
        else:
            f.write('Page {0}\n'.format(page_num+1))
            f.write(''.center(100, '-'))
#             f.write(txt)
    f.close()


def extract_information(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.getDocumentInfo()
        number_of_pages = pdf.getNumPages()

    txt = f"""
    Information about {pdf_path}: 

    Author: {information.author}
    Creator: {information.creator}
    Producer: {information.producer}
    Subject: {information.subject}
    Title: {information.title}
    Number of pages: {number_of_pages}
    """

    print(txt)
    return information


if __name__ == '__main__':
    path = 'sample2.pdf'
    information = extract_information(path)

# pdfminer library focuses entirely on getting and analyzing text data
# The result more refine text extraction


resource_manager = PDFResourceManager()
fake_file_handle = io.StringIO()
converter = TextConverter(
    resource_manager, fake_file_handle, laparams=LAParams())
page_interpreter = PDFPageInterpreter(resource_manager, converter)

with open('80-56-17801-512G_SDCZ430_G46_specs.pdf', 'rb') as fh:

    for page in PDFPage.get_pages(fh,
                                  caching=True,
                                  check_extractable=True):
        page_interpreter.process_page(page)

    text = fake_file_handle.getvalue()

# close open handles
converter.close()
fake_file_handle.close()

print(text)


def parsedocument(document):
    # convert all horizontal text into a lines list (one entry per line)
    # document is a file stream
    lines = []
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(document):
        interpreter.process_page(page)
        layout = device.get_result()
        for element in layout:
            if isinstance(element, LTTextBoxHorizontal):
                lines.extend(element.get_text().splitlines())
    return lines


# regex findall
text = "string"  # some string
re.findall(
    "[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}-[a-zA-Z0-9]{1,10}G", text)  # ['80-56-17801-512G', '80-56-17801-512G']
re.findall("[0-9]{10}", text)  # ['1965917932']
# ['130MB/s', '130MB/s', '130MB/s', '130MB/s', '130MB/s']
re.findall("[0-9]{1,10}MB/s", text)
re.findall("Made in [a-zA-Z1-9]{1,10}", text)  # ['Made in China']


reader = PdfReader("input.pdf")
writer = PdfWriter()


writer.add_page(reader.pages[0])
writer.pages[0].rotate(90)

with open("output.pdf", "wb") as fp:
    writer.write(fp)
