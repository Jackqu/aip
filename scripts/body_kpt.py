from aip import AipBodyAnalysis
APP_ID = '17067353'
API_KEY = 'ovdLIpCXNFAHHYXCASwVrO4u'
SECRET_KEY = 'eyCudNEFKkQtH9KPfvyLr16KMendt1ir'
client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('./test_imgs/body_1.jpg')
kpt_result = client.bodyAnalysis(image)
print(kpt_result)
