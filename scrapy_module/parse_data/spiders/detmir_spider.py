from pathlib import Path

import scrapy


class DetmirSpider(scrapy.Spider):
    name = "detmir"

    def start_requests(self):
        urls = ["https://www.detmir.ru/catalog/index/name/block/brand/7/page/" + str(i) + "/?order=price-asc" for i in range(1, 18)]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        d = {}
        for i, row in enumerate(response.css("div.wV section.wY table tr.V_1")):
            if row.css("span::text").get() in ['Длина упаковки, см', 'Ширина упаковки, см', 'Высота упаковки, см', 'Вес упаковки, кг']:
                d[row.css("span::text").get()] = row.css("td::text").get()

        if len(d) != 4:
            next_pages = response.css("div.hv section a::attr(href)").extract()
            for p_url in next_pages:
                yield scrapy.Request(response.urljoin(p_url),
                                         callback=self.parse)
        else:
            d["Цена"] = response.css("div.wV div.beW div.beY p::text").get()
            print("[ %s | %s ]" % (response.url, d))
            yield d

        
