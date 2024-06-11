


def extract_tag(tag):
    if len(tag) > 5:
        return tag.split("value': '",1)[1][:-3]


def create_tags(review):
    if type(review) == str:
        if 'очеред' in review or 'долго' or 'медлен':
            return 'очередь'
        # кредит, банкомат
        elif 'навяз' in review:
            return 'навязывание_продуктов'
        elif 'сотрудн' in review:
            return 'сотрудники'
        elif 'парков' in review:
            return 'парковка'
        else:
            return None
    else:
            return None

def rename_tags(tag):
    if type(tag) == str:
        if 'навязыван' in tag:
            return 'навязывание_продуктов'
        elif 'благодарность общая' in tag:
            return 'благодарность_общая'
        elif 'комфорт' in tag:
            return 'комфорт'
        elif 'банкомат' in tag:
            return 'банкоматы'
        elif 'очеред' in tag:
            return 'очередь'
        elif 'ВТБ' in tag:
            return 'мобайл/онлайн'
        # elif 'сотрудн' in tag:
        #     return 'сотрудники'
        elif 'график' in tag:
            return 'график_работы'
        elif 'карт' in tag:
            return 'карта'
        elif 'парков' in tag:
            return 'парковка'

        # elif 'благодарность общая' in tag or 'качество обслуживания' in tag or 'кассовое обслуживание' in tag or 'сотрудник' in tag:
        # # elif all(x in tag for x in ['благодарность общая', 'качество обслуживания', 'кассовое обслуживание', 'сотрудник']):
        #     return 'обслуживание сотрудников'
        # elif 'перевод_пенсии' in tag or 'автокредит' in tag or 'вклад' in tag or 'инвестиции' in tag or 'страхование' in tag or 'переводы' in tag:
        # # elif all(x in tag for x in ['перевод_пенсии', 'автокредит', 'вклад', 'инвестиции', 'страхование', 'переводы']):
        #     return 'продукты'

        else:
            return tag
    else:
        return tag

def join_words(text):
    interm =  ','.join(text)
    return interm.replace(",", " ")
