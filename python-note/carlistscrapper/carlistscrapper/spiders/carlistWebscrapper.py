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
    
        next_page = response.css('li.next').css('a').attrib['href']
        print('THIS IS NEXT PAGE', next_page)
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)
