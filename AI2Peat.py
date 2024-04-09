import os
import math
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Constants turned into Variables
network_node_folder = 'C:\\Users\\Pc\\Desktop\\Python Codes\\my_code\\AI2PeatFolder\\'
network_node_filename = 'AI2Peat_List.csv'
network_edge_filename = 'AI2Peat_Relationship.csv'

citation_tag = {'por': "Citado por", 'eng': 'Cited by'} #used for searching in google scholar in Portuguese and English
scraping_language = 'por'
FirstSearch = "Playing atari with deep reinforcement learning"
Search = FirstSearch
SearchBarName = 'q'
TitleXPath = '//div[@class="gs_ri"]/h3/a'
AuthorXPath = '//div[@class="gs_ri"]/div/a'
NextButtonXPath = '//button[@class="gs_btnPR gs_in_ib gs_btn_lrge gs_btn_half gs_btn_lsu"]'
PaperNumber = 'Paper_Number'
PaperName = 'Paper_Name'
SearchSite = 'https://scholar.google.com/'
DriverPath = r"C:\Users\Pc\Desktop\Python Codes"

#Function for creating the CSV files
def init_files():

    node_columns = ['Paper_Number', 'Paper_ID', 'Paper_Name', 'Paper_Link', 'Main_Author', 'Scrapped']
    node_df = pd.DataFrame(columns=node_columns)
    # node_df.set_index('Paper_Number', inplace=True)

    edge_columns = ['Paper_Number_Cited', 'Paper_Number_Quoter']
    edge_df = pd.DataFrame(columns=edge_columns)

    node_df.to_csv(network_node_filename, sep=';', index=False)
    edge_df.to_csv(network_edge_filename, index=False, sep=';')


#Function for opening the browse
def driver_init():
    driver = webdriver.Chrome()
    driver.get(SearchSite)
    driver.implicitly_wait(0.5)
    return driver



#Function for strating the search
def search_init(driver):
    SearchBar = driver.find_element(By.NAME, SearchBarName)
    SearchBar.send_keys(Search)
    SearchBar.send_keys(Keys.ENTER)




#Function for searching the Citation button. It will find the Citations link. It extracts the number of papers from the link, then divides by 10, so we can now how many pages the
#program will need to through to get all the pages. After that it clicks on the lik.
def citationbuttom(driver):
    CitationButtom = driver.find_element(By.PARTIAL_LINK_TEXT, citation_tag[scraping_language])
    PagesList = CitationButtom.text
    PagesSplit = PagesList.split()
    NumberPages = math.ceil(int(PagesSplit[-1]) / 10)
    CitationButtom.click()

    return NumberPages

#Function for writing AI2Peat_List
def add_node(filename, paper_id, paper_title, paper_href, first_author, paper_scraped_already='No'):
    # 1. load the dataframe
    node_df = pd.read_csv(filename, sep=';')

    # 2. does the dataframe already have the paper_title?
    if len(node_df[(node_df['Paper_Name'] == paper_title) & (node_df['Main_Author'] == first_author)]) > 0 :
    #if paper_title in node_df['Paper_Name']:
        # the paper is already present: don't add it anymore, return the paper Paper_Number
        csv_paper_number = list(node_df.loc[(node_df['Paper_Name'] == paper_title) & (node_df['Main_Author'] == first_author), 'Paper_Number'])[0]
        print(f'paper {paper_title} by author {first_author} already exists, csv_paper_number={csv_paper_number}')
        return csv_paper_number # this return statement fetches the (assumed to be) correct value of paper_number

    # the paper does not exist => add it to the dataframe, save it, and return its index (i.e. the last row index)

    added_paper_number = len(node_df)

    new_row = pd.DataFrame([[added_paper_number, paper_id, paper_title, paper_href, first_author, paper_scraped_already]],
                           columns=node_df.columns)
    node_df = pd.concat([node_df, new_row], ignore_index=True)

    node_df.to_csv(network_node_folder + network_node_filename, sep=';', index=False)
    # return the index of the paper just added
    return added_paper_number

#Function for writing AI2Peat_Rlationship
def add_edge(original_index, index):
    edge_df = pd.read_csv(network_node_folder + network_edge_filename, sep=';')
    new_row2 = pd.DataFrame([[original_index, index]], columns=edge_df.columns)
    edge_df = pd.concat([edge_df, new_row2], ignore_index=True)
    edge_df.to_csv(network_node_folder + network_edge_filename, sep=';', index=False)

# Writing into the CSV function
def writtingcsv(driver, OriginalIndex, Index):

    Titles = driver.find_elements(By.XPATH, TitleXPath)
    Authors = driver.find_elements(By.XPATH, AuthorXPath)


    for i in range(len(Titles)):

        Index = add_node(network_node_folder + network_node_filename,
                         Titles[i].get_attribute('id'),
                         Titles[i].text, Titles[i].get_attribute('href'), Authors[i].text)


        add_edge(OriginalIndex, Index)



def scrapping(driver):
    global Index

    NumberPages = citationbuttom(driver)

    # -----------------------------------------------------------------------

    # time.sleep(5)

    # Here it will start the scrapping and write the data into both CSV files created.
    df = pd.read_csv(network_node_folder + network_node_filename, sep=';')
    if len(df) == 1:
        OriginalIndex = Index
    else:
        random_row = papers_df.sample(n=1).iloc[0]
        OriginalIndex = random_row['Paper_Number']
        print('here')



    for PageNext in range(NumberPages):
        writtingcsv(driver, OriginalIndex, Index)

        # If there isn't anymore pages to go through, this try/catch clause will finish the loop. It is needed, because even if the paper
        # has thousands of citations, the google scholar has a limit of 100 pages.
        try:
            NextPage = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable(
                    (By.XPATH, NextButtonXPath))
            )
            NextPage.click()
        except Exception as e:
            print("Failed to click the 'Next' button. Exiting loop.")
            break  # Exit the loop if clicking fails


def scrapping2(driver):
    global Search

    # split this function into 2 functinos, one that samples a paper title from the csv file and one that scrapes given  a paper name

    # function 1: input is a csv file, output is a paper name
    # function 2: input is a paper name output is nothing (but inside the function the csv file is updated)

    # ==========================


    # add another column called scrapped
    # use gitlab
    # make your own github.com account and learn how to use it
    # represent the paper in wa ay the algoirith understand, document embedding. Extract key words from a paper, author, abstract.
    # huggingface.co/sentence-transformers
    # vector maths
    # cosine similarity (check sklearn)

    # we get here and Search contains a paper name

    # In this part it will start getting the data from the other papers. If the data in contained in the column is equal to the first paper we searched
    # or it is the title of the column,it will instead pass and continue the loop
    lastRow = (papers_df['Paper_Number'].iat[-1])

    NumberLoops = int(lastRow)


    for i2 in range(NumberLoops):
        random_row = papers_df.sample(n=1).iloc[0]
        Search = random_row['Paper_Name']


        if papers_df.loc[papers_df['Paper_Name'] == Search, 'Scrapped'].values[0] == 'Yes':

            pass
        # ----------------------------------------------------------
        # Here is where the code will open the Chrome browser again and it will get the name of the paper from the CSV and search for it
        else:

            # ===============================================
            # this below is code that takes in input a paper name and scrapes it by extracting the citations

            papers_df.loc[papers_df['Paper_Name'] == Search, "Scrapped"] = "Yes"
            # Save the DataFrame back to CSV
            papers_df.to_csv(network_node_folder + network_node_filename, sep=';', index=False)



            search_init(driver)

            time.sleep(30)
            OriginalIndex = random_row['Paper_Number']

            # ---------------------------------------------------
            # Here it will find the Citations link. It extracts the number of papers from the link, then divides by 10, so we can now how many pages the
            # program will need to through to get all the pages. After that it clicks on the link. If the paper isn't cited by any other, it will close the window and the loop, and continue
            # searching for toher papers

        #try:
        #    citationbuttom(driver)

        #except Exception as e:

        #    driver.close()  # Close the webpage window
        #    continue

        # ------------------------------------------------------------------------
        # Here it will start the scrapping and write the data into both CSV files created.
        scrapping(driver)


#-------------------------------------------------
#Main Code
Index = 0
#Here is is checking if the CSV file exists or not, if not it creates AI2Peat_List.csv and AI2Peat_Relationship.csv

# the csv file doesnt exist => start scraping from FirstSearch
# paper_title = FirstSearch
# else :
# the csv file exists -> sample a paper title from the csv file
# this below is a function that returns a paper title:
# open csv file (with pandas)
# sample a paper title (with pandas)
# paper_title = random_paper_title()
# outside the if else statement
# scraping_function(paper_title)

# write code to initialise the nodes and edges filenames
if not os.path.exists(network_node_folder + network_node_filename):

    init_files()


#Here it will check if the CSV file is empty, if it is, the program will search for "Playing Atari..."
df = pd.read_csv(network_node_folder + network_node_filename, sep=';')
if len(df) == 0:
    # ---------------------------------------------------------------------
    # In this part, the program will acess the Google Scholar site.

    driver = driver_init()
    # --------------------------------------------------------------------
    # Here it will acess the Search Bar and search for the first paper

    search_init(driver)
    time.sleep(10)
    # -------------------------------------------------------------------
    # Now it will find the name of the first paper by the XPath and then write it on the CSV file and then click on the paper.
    # Index = 0
    OriginalTitle = driver.find_element(By.XPATH, TitleXPath)
    OriginalAuthors = driver.find_elements(By.XPATH, AuthorXPath)
    OriginalName = OriginalTitle.text

    # -----------------------------------------------------------------
    # Here we are opening the CSV file and writing the data from the first paper
    add_node(network_node_folder + network_node_filename,
              OriginalTitle.get_attribute('id'),
             OriginalTitle.text, OriginalTitle.get_attribute('href'),OriginalAuthors[0].text, paper_scraped_already='Yes')

    # --------------------------------------------------------------------
    # Here it will find the Citations link. It extracts the number of papers from the link, then divides by 10, so we can now how many pages the
    # program will need to through to get all the pages. After that it clicks on the link.

    scrapping(driver)
# --------------------------------------------------------------------
# If the CSV is not empty, then the program will sample one of the other papers in the CSV and start searching for it
else:
    papers_df = pd.read_csv(network_node_folder + network_node_filename, sep=';')

    driver = driver_init()


    scrapping2(driver)

    # Search for debuggers
    # SequenceMatcher (difflib)
