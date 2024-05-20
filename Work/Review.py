import pandas as pd

# Load and preprocess the dataset
file_path = 'CCleaner Reviews.xlsx'
columns = ['App', 'App Version Name', 'Reviewer Language',
           'Device', 'Star Rating', 'Review EN']

# Load only the specified columns from Excel file
data = pd.read_excel(file_path, usecols=columns)

# Replace empty values in "App Version Name" column with "24.03.1"
data['App Version Name'].fillna('24.03.1', inplace=True)

# Create a new column named "Category" without assigning any values
data['Category'] = None

# Filter rows where "Review EN" column has a non-null value and Star Rating is equal to or less than 3
filtered_data = data[(data['Review EN'].notnull())
                     & (data['Star Rating'] <= 2)]

# Conditionally assign "Russia" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Reviewer Language'] == 'ru') & (filtered_data['Review EN'].str.contains(
    'politics|work|Russia|region|location|available|banned|working|Russian|sanctions|sanction|leaving|services|service|open|Russophobes|Russophobia|country|downloading|download|political|Useless|Racists|Politicized|supported|off|load|Nazism|laws|Finished',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Russia'

# Conditionally assign "Paid but not Premium" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    ' didn\'t get activated|paid the fee, but I cannot use it|PRO and it won\'t take effect|Where is my premium|proof of my Pro|can\'t activate|didn\'t upgrade|got no membership',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Paid but no Premium'

# Conditionally assign "Files Deletion" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'delete all my gallery|deleted all|deleted my|deleted the images|deleted files|deleted everything|photos are now gone|deleted important photo|lost tons of pictures|remove all my whateApp videos|deleted at least a hundred photos|wiped my entire gallery|all of my photos|videos deleted|photos were unrecoverable|gallery is permanently deleted',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Files Deletion'

# Conditionally assign "Freezes" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'slows the device|slowed down my mobile',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Freezes'

# Conditionally assign "Slows down device" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'stuck at|freezes|freeze|freezing|alway stop',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Slows down device'

# Conditionally assign "Not Simple" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'not simple|app simple?|confusing reading|explanations of how it works|more complicated|difficult to understand',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'Not Simple'

# Conditionally assign "No trial period" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'non tester|no testing|without being able to try|app trials',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = 'No trial period'

# Conditionally assign "Ads don't load" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'Internet is not connected|internet connection|ads do not load|ads dont load|ads don\'t load',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Ads don't load"

# Conditionally assign "Payed App" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'pay for it|payed|not free|photos are gone|Ask for money|pushes you to buy|you have to buy|It is for payment|Money up front|not function for free|no free app|Not functional without an upgrade|don’t offer it as free|No free version|all paid|wants to charge|forced upgrade|features require pay|force upgrade|have to pay|have to pay|requires payment|Asking money|until you pay|cleaning but at a cost|requires money|wants payment|everything has to pay|forces you to pay|Only viewing is free|asking for money|everything is paid|free to download|forced you to upgrade|pay for everything',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Payed app"

# Conditionally assign "Permissions again" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'why are there still ads|subscription and still get a lot of advertisements|never removed the ads',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Cross-Promotion while Premium"

# Conditionally assign "Ads" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'annoying ads|too much ads|intrusive ads|Continuous advertising|lot of propaganda|overwhelming amount of ads|functions require you to view|commercials|LONG ADVERTS|endless ads|has ads|Just ads|Ads are ridiculous|need to pay|one advertisement after another|ADVERTISING ONLY|just ads now|minefield of adverts|bothersome ads|unnecessary ads|Too many adds|too many ads|too much advertising|just propaganda|several ads|gained advertising|annoyance of the ads|more ads',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Ads"

# Conditionally assign "Crashes" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'closes by itself|crash|crashes', case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Crashes"

# Conditionally assign "Free version useless" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'useless if you don\'t purchase|Without a subscription you can\'t|functions only for a fee|Nothing free here|full cleaning after payment only|everything has to be used with Premium|pay to unlock features|Useless without buying pro|paid subscription is forced|behind a pay wall|Without the pro version|locked in premium|pushing the paid service|locked behind a pay wall|YOU NEED TOBL PAY|subscription for it|useless if you dont purchase|without premium|little the application deletes|free features are already present in the phone|almost useless|unless we upgrade|pay rail|useless if you do not purchase|Useless app|without paying for the pro license|Less and fewer options for us to pay|behind a paywall|deep cleaning is a premium|free version is very limited|Pay or it doesn\'t',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Free version useless"

# Conditionally assign "Doesn't Clean" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'doesnt clean|does not clean|Doesn\'t Clean|cleans anything|useless|don\'t cleaning|does not delete anything|none of these files was deleted|no longer works|clean anything|doesn\'t even work|memory is ALWAYS in negative territory|didnt clear|it works or not|same space show after cleaning|Doesn\'t do a good job cleaning|doesnt work|does not work|Doesn\'t Work|don\'t work',
    case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Doesn't Clean"

# Conditionally assign "Permissions again" to "Category" column based on criteria
filtered_data.loc[(filtered_data['Review EN'].str.contains(
    'permissions again', case=False, na=False)) & (filtered_data['Category'].isnull()), 'Category'] = "Permissions again"

print(filtered_data.head())

data.update(filtered_data)
# Specify the file path for the exported Excel file
export_file_path = 'filtered_reviews.xlsx'

# Export the filtered DataFrame to Excel
data.to_excel(export_file_path, index=False)

print("Filtered data exported successfully to:", export_file_path)
