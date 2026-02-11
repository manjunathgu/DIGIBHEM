# IPL Exploratory Data Analysis (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    file_path = "dataset/matches.csv"
    try:
        # Manually specify column names if the dataset lacks headers
        column_names = [
            'id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
            'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
            'win_by_wickets', 'player_of_match', 'venue', 'umpire1', 'umpire2', 'umpire3'
        ]
        data = pd.read_csv(file_path, header=None, names=column_names)
        print("Dataset loaded successfully with manually specified column names.")
        print("\nColumn names in the dataset:")
        print(data.columns)  # Debugging step to verify column names
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None

# Data Cleaning
def clean_data(data):
    print("\nChecking for missing values...")
    print(data.isnull().sum())

    # Drop rows only if critical columns are missing
    critical_columns = ['team1', 'team2', 'winner', 'player_of_match', 'venue']
    data = data.dropna(subset=critical_columns)
    print("\nDropped rows with missing values in critical columns.")

    # Debugging step: Print the first few rows of the dataset after cleaning
    print("\nDataset after cleaning:")
    print(data.head())

    # Rename columns for better usability (if needed)
    data.rename(columns={
        'team1': 'Team1',
        'team2': 'Team2',
        'winner': 'Winner',  # Ensure this matches the dataset
        'player_of_match': 'Player_of_Match',
        'toss_decision': 'Toss_Decision',
        'venue': 'Venue'
    }, inplace=True)

    return data

# Exploratory Data Analysis
def analyze_data(data):
    # Most successful teams
    print("\nMost Successful Teams:")
    team_wins = data['winner'].value_counts()
    print(team_wins.head(5))

    # Top-performing players
    print("\nTop Players by Player of the Match Awards:")
    top_players = data['player_of_match'].value_counts().head(5)
    print(top_players)

    # Toss decision impact
    print("\nToss Decision Impact:")
    toss_impact = data['Toss_Decision'].value_counts()
    print(toss_impact)

    # Win distribution by venue
    print("\nWin Distribution by Venue:")
    venue_wins = data['Venue'].value_counts().head(5)
    print(venue_wins)

    return team_wins, top_players, toss_impact, venue_wins

# Visualizations
def create_visualizations(team_wins, top_players, toss_impact, venue_wins):
    # Check if the data is empty before plotting
    if team_wins.empty or top_players.empty or toss_impact.empty or venue_wins.empty:
        print("\nError: No data available for visualization. Please check the dataset.")
        return

    # Bar chart for most successful teams
    plt.figure(figsize=(10, 6))
    team_wins.head(5).plot(kind='bar', color='skyblue')
    plt.title('Top 5 Most Successful Teams')
    plt.xlabel('Teams')
    plt.ylabel('Total Wins')
    plt.show()

    # Bar chart for top players
    plt.figure(figsize=(10, 6))
    top_players.plot(kind='bar', color='orange')
    plt.title('Top 5 Players by Player of the Match Awards')
    plt.xlabel('Players')
    plt.ylabel('Awards')
    plt.show()

    # Toss decision impact
    plt.figure(figsize=(10, 6))
    toss_impact.plot(kind='bar', color='green')
    plt.title('Toss Decision Impact')
    plt.xlabel('Decision')
    plt.ylabel('Count')
    plt.show()

    # Win distribution by venue
    plt.figure(figsize=(10, 6))
    venue_wins.plot(kind='bar', color='purple')
    plt.title('Top 5 Venues by Win Count')
    plt.xlabel('Venue')
    plt.ylabel('Wins')
    plt.show()

# Main function
def main():
    data = load_data()
    if data is not None:
        clean_data_ = clean_data(data)
        # Debugging step: Print the first few rows of the dataset after cleaning
        print("\nDataset after cleaning:")
        print(clean_data_.head())

        # Ensure the dataset is not empty
        if clean_data_.empty:
            print("\nError: The dataset is empty after cleaning. Please check the data.")
            return None

        team_wins, top_players, toss_impact, venue_wins = analyze_data(clean_data_)
        create_visualizations(team_wins, top_players, toss_impact, venue_wins)

if __name__ == "__main__":
    main()