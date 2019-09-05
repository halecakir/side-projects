import scrapy

class AptoideSpider(scrapy.Spider):
    name = "aptoide-spider"
    COUNT_MAX = 500000

    count = 0

    def start_requests(self):
        urls = ["https://en.aptoide.com/apps/latest/more"]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parseLatestApps)

    def parseLatestApps(self, response):
        #Application links
        app_pages = response.xpath('//*[@class="bundle-item__info__span bundle-item__info__span--big"]/a/@href').extract()
        next_page = response.xpath('//*[@class="aptweb-button widget-pagination__next"]/a/@href').extract_first() 
        for link in app_pages:
            if self.count < AptoideSpider.COUNT_MAX:
                self.count += 1
                yield scrapy.http.Request(url=link, callback=self.parseApp)
        if self.count < AptoideSpider.COUNT_MAX:
            yield scrapy.Request(url=next_page, callback=self.parseLatestApps)

    def parseApp(self, response):
        app_name = response.xpath('//*[@class="header__title"]/text()').extract_first()
        app_description = response.css('div.view-app__description *::text').extract()
        app_permissions = response.xpath('//*[@class="app-permissions__row"]/span/text()').extract()
        response.xpath('//*[@class="aptweb-button widget-pagination__next"]/a/@href').extract()
        
        
        yield {
                'name': app_name,
                'description': "\n".join([d.strip() for d in app_description]),
                'permission': [p.split(".")[-1] for p in app_permissions if p.startswith("android.permission")],
                'url': response.request.url,
            }


