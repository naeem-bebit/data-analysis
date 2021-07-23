import scrapy

class carlistWebscrapper(scrapy.Spider):
    name = 'carlist'
    start_urls = ['https://www.carlist.my/cars-for-sale/malaysia?inspected=true']

    def parse(self, response):
        for carlist in response.css('article.listing'):
            try:
                yield {
                    'car':carlist.css('div.listing__rating-model::text').get(),
                    'price':carlist.css('div.listing__price::text').get().replace('RM ',''),
                    'link':carlist.css('a.listing__overlay').attrib['href'],
                }
            except:
                yield {
                    'car':carlist.css('div.listing__rating-model::text').get(),
                    'price':carlist.css('div.listing__price::text').get().replace('RM ',''),
                    'link':carlist.css('a.listing__overlay').attrib['href'],
                }
    
    def pagination(self, response):
        for pagination in response.css('div.pagination--footer'):
            try:
                pagination.css('a').attrib['href']
            except:
                pass
