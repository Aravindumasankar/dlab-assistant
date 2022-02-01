from google_images_search import GoogleImagesSearch

# if you don't enter api key and cx, the package will try to search
# them from environment variables GCS_DEVELOPER_KEY and GCS_CX
gis = GoogleImagesSearch('AIzaSyC1YPj2vXD9_r1FUOE80fB1XvVJNz5ixQo', '001888839177267368276:l53xanw83ca')

# example: GoogleImagesSearch('ABcDeFGhiJKLmnopqweRty5asdfghGfdSaS4abC', '012345678987654321012:abcde_fghij')

#define search params:
_search_params = {
    'q': '90s kids memes',
    'searchType': 'image',
    'start': 21
}

# # this will only search for images:
gis.search(search_params=_search_params)
#
# this will search and download:
# gis.search(search_params=_search_params, path_to_dir='../memes/dataset/90s kids/')
#
# # # this will search, download and resize:
# # gis.search(search_params=_search_params, path_to_dir='dataset/', width=500, height=500)
#
#search first, then download and resize afterwards
# gis.search(search_params=_search_params)
# for image in gis.results():
#     image.download('dataset/90s Kids')