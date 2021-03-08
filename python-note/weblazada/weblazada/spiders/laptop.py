import scrapy


class LaptopSpider(scrapy.Spider):
    name = 'laptop'
    # allowed_domains = ['x']
    # start_urls = ['http://x/']

    def start_requests(self):
        url = "https://www.lazada.com.my/catalog/?q=laptop&_keyori=ss&from=input&spm=a2o4k.home.search.go.75f82e7eZpzVGx"
        yield scrapy.Request(url)


    def parse(self, response):
        products_selector = response.css('[data-tracking="product-card"]')
        for product in products_selector:
            yield {
                'name': product.css('a[title]::attr(title)').get(),
                'price': product.css('span:contains("RM")::text').get()
            }
