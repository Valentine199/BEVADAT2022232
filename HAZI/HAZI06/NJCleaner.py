import pandas as pd


class NJCleaner():
    def __init__(self, csv_path: str) -> None:
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self):
        order = self.data.sort_values(by=['scheduled_time'])
        return order

    def prep_df(self, path='data/NJ.csv'):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()

        NJCleaner.save_first_60k(self, path)

        ##self.data.to_csv(save_csv_path)

    def drop_columns_and_nan(self):
        dropped = self.data.drop(['from', 'to'], axis=1)
        dropped = dropped.dropna()
        return dropped

    def convert_date_to_day(self):
        day_transformed = self.data.copy()
        day_transformed['date'] = pd.to_datetime(day_transformed['date'])
        day_transformed['day'] = day_transformed['date'].dt.day_name()
        day_transformed = day_transformed.drop(['date'], axis=1)
        return day_transformed

    def convert_scheduled_time_to_part_of_the_day(self):
        schedule = self.data.copy()
        schedule['scheduled_time'] = pd.to_datetime(schedule['scheduled_time'])
        schedule = schedule.set_index('scheduled_time')

        schedule['part_of_the_day'] = ""

        schedule.loc[schedule.between_time('4:00', '7:59').index, 'part_of_the_day'] = 'early_morning'
        schedule.loc[list(schedule.between_time('8:00', '11:59').index), 'part_of_the_day'] = 'morning'
        schedule.loc[schedule.between_time('12:00', '15:59').index, 'part_of_the_day'] = 'afternoon'
        schedule.loc[schedule.between_time('16:00', '19:59').index, 'part_of_the_day'] = 'evening'
        schedule.loc[schedule.between_time('20:00', '23:59').index, 'part_of_the_day'] = 'night'
        schedule.loc[schedule.between_time('0:00', '3:59').index, 'part_of_the_day'] = 'late_night'

        schedule.reset_index(drop=True, inplace=True)
        #schedule.drop(['scheduled_time'], axis=1, inplace=True)
        return schedule

    def convert_delay(self):
        delayes = self.data.copy()
        delayes['delay'] = 0
        delayes.loc[delayes['delay_minutes'] >= 5, 'delay'] = 1

        return delayes

    def drop_unnecessary_columns(self):
        droppos = self.data.copy()
        droppos.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1, inplace=True)

        return droppos

    def save_first_60k(self, path):
        to_print = self.data.iloc[:60000]

        to_print.to_csv(path, index=False)







#cleaned = NJCleaner('2018_03.csv')
#cleaned.prep_df('save.csv')




