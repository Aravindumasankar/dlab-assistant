from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': 'dataset/sowcar janaki'})
filters = dict()
google_crawler.crawl(keyword='sowcar janaki black and white photos', filters=filters, offset=0, max_num=30,
                     min_size=(200,200), max_size=None, file_idx_offset=0)

bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': 'dataset/sowcar janaki'})
bing_crawler.crawl(keyword='sowcar janaki', filters=None, offset=0, max_num=20)

# baidu_crawler = BaiduImageCrawler(storage={'root_dir': 'dataset/vadivelu'})
# baidu_crawler.crawl(keyword='vadivelu', offset=0, max_num=1000,
#                     min_size=(200,200), max_size=None)