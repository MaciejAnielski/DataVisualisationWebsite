import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os



# Create global save_plot function:

def save_plot(plot_title):

	plt.savefig(f'plots/{plot_title}', dpi=300, bbox_inches='tight')
	
class CreateDirectory:

    def __init__(self):
        
        os.makedirs('data', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

# Creating the USPresidentialApprovalProcessor object which includes methods related to importing, processing and graphing US Presidential Approval

class USPresidentialApprovalProcessor:

    def __init__(self):
        
        # These are the data sources for the approval data.
        
        self.approval_data_urls = ['https://projects.fivethirtyeight.com/polls/data/president_approval_polls.csv',
        'https://projects.fivethirtyeight.com/polls/data/president_approval_polls_historical.csv']   
        
    
    def import_data(self):
        
        approval_data = pd.DataFrame()
        
        for i in range(len(self.approval_data_urls)):
            
            temp_df = requests.get(self.approval_data_urls[i]).text
            temp_df = pd.read_csv(StringIO(temp_df))
            
            approval_data = pd.concat([approval_data, temp_df], ignore_index = True)
            
        approval_data['end_date'] = pd.to_datetime(approval_data['end_date'], format='%m/%d/%y')
            
        approval_data['net_approval'] = (approval_data['yes'] - approval_data['no'])
        
        approval_data.to_csv('data/us_presidentail_approval.csv')
        
        return(approval_data)
            
    
    def plot_approval_2017_present(self):
        
        approval_data = self.import_data()
        
        # Create plot data
        
        plot_data = approval_data.groupby(pd.Grouper(key = 'end_date', freq = '1W'), as_index=False).agg({'end_date': 'first', 'net_approval': 'mean'}).sort_values('end_date')
        
        # Prepare data for plotting.
        
        x_values = plot_data['end_date']
        y_values = plot_data['net_approval']
    	
    	# Add presidential term data.
    	
        inauguration_data = {
            'president' : ['Trump','Biden','Trump'],
            'inauguration_date' : ['2017-01-20','2021-01-20','2025-01-20']}
        
        inauguration_data = pd.DataFrame(inauguration_data, columns = ['president', 'inauguration_date'])
        inauguration_data['inauguration_date'] = pd.to_datetime(inauguration_data['inauguration_date'])
    	
        plot_data['president'] = np.where(
            (plot_data['end_date'] >= pd.to_datetime('2017-01-20')) &
            (plot_data['end_date'] < pd.to_datetime('2021-01-20')),
            'Trump First Term',
            np.where(
                (plot_data['end_date'] >= pd.to_datetime('2021-01-20')) & 
                (plot_data['end_date'] < pd.to_datetime('2025-01-20')),
                'Biden',
                'Trump Second Term'
            )
        )
    	
    	# Create plot.
    	
        fig, ax = plt.subplots(facecolor = '#f2f0ef', figsize = (12,6))
        ax.set_facecolor('#f2f0ef') 
    	
        plt.axhline(y=0, color='black', linestyle='-', linewidth = 0.7)
    	
    	# Create new plot for each presidential term with appropriate colours.
	
        for president, color in {'Trump First Term': 'red', 'Biden': 'blue', 'Trump Second Term': 'red'}.items():
            mask = plot_data['president'] == president
            plt.plot(plot_data.loc[mask, 'end_date'], 
                     plot_data.loc[mask, 'net_approval'],
                     color=color,
                     label=president,
                     linewidth = 3)
    		         
     	# Add inauguration date labels.
	
        for idx, row in inauguration_data.iterrows():
            plt.vlines(x=row['inauguration_date'], ymin = min(y_values - 5), ymax = max(y_values + 5), linestyle='--', color = 'black')
            plt.text(row['inauguration_date'], max(y_values + 5), f'{row['president']}\nInauguration', ha='center', va='bottom', rotation=0)
    		
		# Add plot details.
	
        plt.title('USA Presidential Approval From Jan 2017 to Present')
        plt.xlabel('Date')
        plt.ylabel('Presidential Approval %')
        plt.ylim(-30,30)
    	
		# Clean up x axis.
	
        plt.xticks(pd.date_range(start = min(x_values), end = max(x_values), freq = '6ME'))
    	
    	# Set the date format on the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y')) 
    	
    	# Remove box around plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    	             
        plt.tight_layout()
        
        plt.savefig('plots/us_presidential_net_approval_2017_present.png', dpi=300, bbox_inches='tight')
    	
    	
    def plot_approval_current(self):
	    
        approval_data = self.import_data()
        
        plot_data = approval_data[approval_data['end_date'] >= pd.to_datetime('2025-01-20')]
        
        x_values_scatter = plot_data['end_date']
        y_values_scatter = plot_data['net_approval']
        
        df_line_data = plot_data.groupby('end_date', as_index = False).mean('net_approval')
        
        x_values_line = df_line_data['end_date']
        y_values_line = df_line_data['net_approval'].rolling(window = 7, min_periods = 1, center=True).mean()
        
        # Create plot.
        
        fig, ax = plt.subplots(facecolor = '#f2f0ef', figsize = (12,6))
        ax.set_facecolor('#f2f0ef') 
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth = 0.7)
        
        plt.scatter(x_values_scatter, y_values_scatter, color = 'red', alpha = 0.1)
        plt.plot(x_values_line, y_values_line, color = 'red', linewidth = 3)
        
        
        # Add plot details.
        
        plt.title('USA Presidential Approval Current Term')
        plt.xlabel('Date')
        plt.ylabel('Presidential Approval %')
        plt.ylim(-30,30)
        
        # Clean up x axis.
        
        plt.xticks(pd.date_range(start = min(x_values_scatter), end = max(x_values_scatter), freq = '1W'))
        
        # Set the date format on the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %y')) 
        
        # Remove box around plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                     
        plt.tight_layout()
        
        plt.savefig('plots/us_presidential_net_approval_current_term.png', dpi=300, bbox_inches='tight')

    	

# Creating the UKVotingIntentionProcessor object which includes methods related to importing, processing and graphing UK Voting Intention Polling Data.

class UKVotingIntentionProcessor:
    def __init__(self):
        self.election_years = {
            'next': {'index': 0, 'date_col': 'Dates conducted', 'years': ['2025', '2024'], 'keep_dfs': [1,2]},
            '2024': {'index': 1, 'date_col': 'Dates conducted', 'years': ['2024', '2023', '2022', '2021', '2020'], 'keep_dfs': [0,4]},
            '2019': {'index': 2, 'date_col': 'Dates conducted', 'years': ['2019', '2018', '2017'], 'keep_dfs': [0,2]},
            '2017': {'index': 3, 'date_col': 'Date(s) conducted', 'years': ['2017', '2016', '2015'], 'keep_dfs': [0,2]},
            '2015': {'index': 4, 'date_col': 'Date(s) conducted', 'years': ['2015', '2014', '2013', '2012', '2011', '2010'], 'keep_dfs': [0,5]},
            '2010': {'index': 5, 'date_col': 'Date(s) Conducted', 'years': ['2010', '2009', '2008', '2007', '2006', '2005'], 'keep_dfs': [0,5]},
            '2005': {'index': 6, 'date_col': 'Dates conducted', 'years': ['2005', '2004', '2003', '2002', '2001'], 'keep_dfs': [0,4]},
            '2001': {'index': 7, 'date_col': 'Date(s) conducted', 'years': ['2001', '2000', '1999', '1998', '1997'], 'keep_dfs': [0,4]},
            '1997': {'index': 8, 'date_col': 'Survey end date', 'years': ['1997', '1996', '1995', '1994', '1993', '1992'], 'keep_dfs': [0,5]},
            '1992': {'index': 9, 'date_col': 'Survey end date', 'years': ['1992', '1991', '1990', '1989', '1987'], 'keep_dfs': [0,4]},
            '1987': {'index': 10, 'date_col': 'Survey end date', 'years': ['1987', '1986', '1985', '1984', '1983'], 'keep_dfs': [0,4]},
            '1983': {'index': 11, 'date_col': 'Survey end date', 'years': ['1983', '1982', '1981', '1980', '1979'], 'keep_dfs': [0,4]},
            '1979': {'index': 12, 'date_col': 'Survey end date', 'years': ['1979', '1978', '1977', '1976', '1975', '1974'], 'keep_dfs': [0,5]},
            # 1974 is missing because of an error in the html parsing that I haven't been able to solve.
            # '1974': {'index': 13, 'date_col': 'Dates conducted', 'years': ['1974', '1974', '1973', '1972', '1971', '1970'], 'keep_dfs': [0,5]},
            '1970': {'index': 13, 'date_col': 'Fieldwork', 'years': ['1970', '1969', '1968', '1967', '1966'], 'keep_dfs': [0,4]},
            '1966': {'index': 14, 'date_col': 'Fieldwork', 'years': ['1966', '1965', '1964'], 'keep_dfs': [0,2]},
            '1964': {'index': 15, 'date_col': 'Fieldwork', 'years': ['1964', '1963', '1962', '1961', '1960', '1959'], 'keep_dfs': [0,5]},
            # Need to fix date conversion because the formats here are a lot different. Sometimes use full month names.
            '1959': {'index': 16, 'date_col': 'Survey End Date', 'years': ['1959', '1958', '1957', '1956', '1955'], 'keep_dfs': [0,4]},
            '1955': {'index': 17, 'date_col': 'Date(s) Conducted', 'years': ['1955', '1954', '1953', '1952', '1951'], 'keep_dfs': [0,4]},
            # Date Column Names in the next tables have different headings for some reason so won't work with current code.
            '1951': {'index': 18, 'date_col': 'Date(s) conducted/ published', 'years': ['1951', '1950'], 'keep_dfs': [0,1]},
            '1950': {'index': 19, 'date_col': 'Date(s) conducted/ published', 'years': ['1950', '1949', '1948', '1947', '1946'], 'keep_dfs': [0,4]},
            # 1945 doesn't work because it groups by year ranges rather than years.
            }
            
    # This method constructs a list of urls to scrape data from.
            
    def get_urls(self):
        
        for year in self.election_years.keys():
            
            if year == '1974':
                self.election_years[year]['url'] = 'https://en.wikipedia.org/wiki/Opinion_polling_for_the_{year}_United_Kingdom_general_elections'
                
            else:
                self.election_years[year]['url'] = f'https://en.wikipedia.org/wiki/Opinion_polling_for_the_{year}_United_Kingdom_general_election'
                
        return(self.election_years) 
        
    # This method scrapes relevant table content after urls constructed.
        
    def get_wiki_tables(self):
        
        self.election_years = self.get_urls()
        
        column_mapping = {
            
            'Date(s) conducted': 'Dates conducted',
            'Date(s) Conducted': 'Dates conducted',
            'Survey end date': 'Dates conducted',
            'Fieldwork': 'Dates conducted',
            'Survey End Date': 'Dates conducted',
            'Date(s) conducted/ published': 'Dates conducted',
            
            'Lib Dems': 'Lib',
            'Lib Dem': 'Lib',
            'Lib. Dems': 'Lib',
            'LD': 'Lib',
            'All': 'Lib',
            
            'Green': 'Grn',
            
            'Reform': 'Ref',
            
            'Con.': 'Con',
            
            'Lab.': 'Lab',
            
            'Ref.': 'Ref',
            
            'Brexit': 'Ref',
            
        }
        
        for year in self.election_years.keys():

            # Get the page content
            response = requests.get(self.election_years[year]['url'])
            soup = BeautifulSoup(response.content, 'html.parser')
        
            # Clean the HTML table by removing problematic attributes
            for tag in soup.find_all(['td', 'th']):
                # Convert string rowspan/colspan to integers
                if tag.has_attr('rowspan'):
                    try:
                        tag['rowspan'] = int(tag['rowspan'].strip('"'))
                    except ValueError:
                        del tag['rowspan']
                if tag.has_attr('colspan'):
                    try:
                        tag['colspan'] = int(tag['colspan'].strip('"'))
                    except ValueError:
                        del tag['colspan']
        
            # Now read the cleaned table
            
            start_index = self.election_years[year]['keep_dfs'][0]
            end_index = self.election_years[year]['keep_dfs'][1] + 1
            
            tables = pd.read_html(StringIO(str(soup)), header = 0)
            
            tables = [table for table in tables if len(table.columns) > 4 and self.election_years[year]['date_col'] in table.columns][start_index:end_index]
            
            tables = [table.assign(**{'date_year_appended': lambda x, y=yr: x[self.election_years[year]['date_col']].astype(str) + ' ' + y}) for table, yr in zip(tables, self.election_years[year]['years'][:len(tables)])]
            
            for i in range(len(tables)):
                
                tables[i] = tables[i].rename(columns = column_mapping)
            
            tables = pd.concat(tables, ignore_index = True).drop(index = 0)
            
            self.election_years[year]['voting_intention'] = tables
            
        return(self.election_years)
        
    # This method cleans and combines tables from scraping.
        
    def clean_wiki_tables(self):
        
        self.election_years = self.get_wiki_tables()
        
        for year in self.election_years.keys():
            
            self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['date_year_appended'].str.replace('-', '–')
            
            if (self.election_years[year]['voting_intention']['end_date'].str.contains('–', na=False).any()):
            
                self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['end_date'].str.split('–', expand=True)[1]
                
                self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['end_date'].fillna(self.election_years[year]['voting_intention']['date_year_appended'])
            
            else:
                pass
            
            self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['end_date'].str.strip()
            
            # This is here because some of the dates come with years already attached and this removes the unnecessary year.
            
            self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['end_date'].apply(lambda x: x[:-5] if len(x) > 13 else x)
            
            self.election_years[year]['voting_intention']['end_date'] = self.election_years[year]['voting_intention']['end_date'].apply(lambda x: f'01 {x}' if len(x) < 9 else x)
            
            self.election_years[year]['voting_intention']['end_date'] = pd.to_datetime(
                pd.to_datetime(
                    self.election_years[year]['voting_intention']['end_date'], 
                    format='%d %b %Y', 
                    errors='coerce'
                ).fillna(
                    pd.to_datetime(
                        self.election_years[year]['voting_intention']['end_date'], 
                        format='%d %B %Y', 
                        errors='coerce'
                    )
                )
            )
            
            self.election_years[year]['voting_intention'].to_csv(f'data/uk_{year}_general_election_voting_intention.csv', index = False)
            
        return(self.election_years)         
            
            
    # This method shows the number of 
        
    def info(self):
        
        self.election_years = self.clean_wiki_tables()
            
        for year in self.election_years.keys():
            
            date_conversion_fail = self.election_years[year]['voting_intention'][self.election_years[year]['Dates conducted']].notnull().sum() - self.election_years[year]['voting_intention']['end_date'].notnull().sum()
            
            date_total = self.election_years[year]['voting_intention'][self.election_years[year]['Dates conducted']].notnull().sum()
            
            print(f'For {year} GE Polls: {date_conversion_fail} out of {date_total} failed.')
            
    def plot_voting_intention(self, ukvi_data, election_year, save_plot = True):
    
        def select_columns(df, columns_list = ['end_date', 'Lab', 'Con', 'Lib', 'Ref', 'UKIP']):
            
            return df.filter(items=columns_list)
        
        self.election_years = ukvi_data
        
        if isinstance(election_year, list):
            
            if len(election_year) > 1:
            
                plot_suffix = f'{election_year[0]}_{election_year[-1]}'
                
            else:
                
                plot_suffix = f'{election_year[0]}'
            
            plot_data = pd.DataFrame()
            
            for year in election_year:
                
                plot_data = pd.concat([plot_data, self.election_years[year]['voting_intention']], ignore_index = True) 
                
        else:
            
            plot_suffix = f'{election_year}'
            
            plot_data = self.election_years[election_year]['voting_intention']
        
        # Select and Clean Data for Graph
        
        plot_data = select_columns(plot_data)
        
        # plot_data['SNP'] = plot_data['Others'].str.split('SNP').str[0].str.strip()
    
        party_columns = plot_data.columns[1:].tolist()
        
        party_colours = {
            
            'Lab': '#d50000',
            'Con': '#0087dc',
            'Ref': '#00bed6',
            'Lib': '#FDBB30',
            'UKIP': '#6D3177'
        }
        
        for party in party_columns:
            
            plot_data[party] = plot_data[party].str.strip('%')
            
            plot_data[party] = pd.to_numeric(plot_data[party], errors='coerce')
            
        # print(plot_data['end_date'].min())
        
        plot_data = plot_data.sort_values(by='end_date')
        
        # plot_data.to_csv('data/plot_data.csv', index = False)
        
        # .dropna(axis = 0)
        
        # Create Graph
        
        fig, ax = plt.subplots(facecolor = '#f2f0ef', figsize = (12,6))
        ax.set_facecolor('#f2f0ef') 
        
        for i in range(len(party_columns)):
            
            x = plot_data['end_date']
            y = plot_data[party_columns[i]]
            
            plt.scatter(x, y, color = party_colours[party_columns[i]], alpha = 0.01)
            
            
        for i in range(len(party_columns)):
            
            x = plot_data['end_date']
            y = plot_data[party_columns[i]]
            
            y_smooth = y.rolling(window=14, min_periods=1, center=True).mean()
            
            plt.plot(x, y_smooth, color = party_colours[party_columns[i]], linewidth = 3, label = party_columns[i])
            
            
        plt.title('UK General Election Voting Intention')
        plt.xlabel('Date')
        plt.ylabel('Voting Intention %')
        
        
        plt.legend(frameon = False, framealpha = 0.0)
        
        plt.ylim(0,60)
        
        # plt.xticks(pd.date_range(start = min(plot_data['end_date']), end = max(plot_data['end_date']), freq = '1M'))
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        
        # Remove box around plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        plt.tight_layout()
        
        
        
        plt.savefig(f'plots/uk_ge_voting_intention_{plot_suffix}.png', dpi=300, bbox_inches='tight')