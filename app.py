import streamlit as st
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, teamgamelog, commonteamroster
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Players'

# App title and navigation
st.title('NBA Prediction App 🏀')
st.session_state.page = st.radio('Select Page:', ['Players', 'Team'])

# Get all NBA teams with retry logic
def get_nba_teams():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            nba_teams = teams.get_teams()
            if nba_teams:
                return nba_teams
            time.sleep(1)  # Add delay between retries
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch NBA teams: {str(e)}")
                return []
            st.warning(f"Retrying to fetch NBA teams (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)  # Add delay between retries
    return []

def get_team_roster(team_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id)
            df = roster.get_data_frames()[0]
            return [{'full_name': row['PLAYER'], 'id': row['PLAYER_ID']} for _, row in df.iterrows()]
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch roster: {str(e)}")
                return []
            st.warning(f"Retrying to fetch roster (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)  # Add delay between retries
    return []

def get_player_stats(player_id, last_n_games=10):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id)
            df = gamelog.get_data_frames()[0].head(last_n_games)
            return df
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch player stats: {str(e)}")
                return None
            time.sleep(1)  # Add delay between retries
    return None

def get_team_stats(team_id, last_n_games=10):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            gamelog = teamgamelog.TeamGameLog(team_id=team_id)
            df = gamelog.get_data_frames()[0].head(last_n_games)
            return df
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to fetch team stats: {str(e)}")
                return None
            time.sleep(1)  # Add delay between retries
    return None

def get_prediction(prompt):
    try:
        response = client.chat.completions.create(
            model="o3-mini-2025-01-31",
            messages=[
                {"role": "system", "content": """
                You are an expert NBA analyst with deep knowledge of basketball analytics, player statistics, and team performance. 
                Your task is to analyze historical data, recent performance trends, and contextual factors to provide a precise prediction.
                
                In your analysis, please consider:
                - Detailed statistics from the last 10 games (including points, rebounds, assists, shooting efficiency, etc.).
                - Trends and consistency in player/team performance, including variance and momentum.
                - Contextual factors such as Home/Away advantage, recent injuries, rivalry dynamics, breaking news, and weather conditions.
                - Comparison of the current prediction value against historical averages and opponent defensive/offensive metrics.
                
                Format your response exactly as follows:
                "OVER:" or "UNDER:" followed by a brief, data-driven explanation that justifies your prediction.
                """},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting prediction: {str(e)}"

# Fetch NBA teams
nba_teams = get_nba_teams()
if not nba_teams:
    st.error("Could not fetch NBA teams. Please try refreshing the page.")
    st.stop()

team_names = [team['full_name'] for team in nba_teams]

if st.session_state.page == 'Players':
    st.header('Player Predictions')
    
    # Team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select Team 1:', team_names, key='team1')
    with col2:
        team2 = st.selectbox('Select Team 2:', team_names, key='team2')
    
    # Get players from selected teams
    team1_id = next(team['id'] for team in nba_teams if team['full_name'] == team1)
    team2_id = next(team['id'] for team in nba_teams if team['full_name'] == team2)
    
    # Get team rosters
    team1_players = get_team_roster(team1_id)
    team2_players = get_team_roster(team2_id)

    if not team1_players:
        st.error(f"Could not fetch roster for {team1}")
        st.stop()
    if not team2_players:
        st.error(f"Could not fetch roster for {team2}")
        st.stop()

    team1_names = [p['full_name'] for p in team1_players]
    team2_names = [p['full_name'] for p in team2_players]
    
    # Player selection
    available_players = team1_names + team2_names
    if not available_players:
        st.error("No players available for the selected teams. Please try different teams.")
        st.stop()
        
    selected_player = st.selectbox('Select Player:', available_players)
    
    # Category selection
    category = st.selectbox('Select Category:', 
                          ['Points', 'Rebounds', 'Assists', '3-Pointers Made', 'Pt\'s + Rebs + Asts'])
    
    # Prediction value input
    prediction_value = st.number_input('Enter prediction value:', min_value=0.0, step=0.5)
    
    # Variables selection
    st.subheader('Additional Variables')
    home_away = st.checkbox('Home/Away Advantage')
    injuries = st.checkbox('Check for Injuries')
    rivalry = st.checkbox('Rivalry Factor')
    news = st.checkbox('Recent News/Scandals')
    weather = st.checkbox('Extreme Weather')
    
    if st.button('Predict'):
        with st.spinner('Fetching player statistics...'):
            try:
                # Find player info from the combined roster
                player_info = next(p for p in (team1_players + team2_players) if p['full_name'] == selected_player)
                stats = get_player_stats(player_info['id'])
            except StopIteration:
                st.error("Player not found. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
                
            if stats is not None:
                prompt = f"""
                Analyze the following scenario for an NBA prediction:
                Player: {selected_player}
                Category: {category}
                Prediction Value: {prediction_value}
                Recent Stats: {stats.to_dict()}
                
                Consider these factors:
                - Home/Away: {home_away}
                - Injuries: {injuries}
                - Rivalry: {rivalry}
                - News/Scandals: {news}
                - Weather: {weather}
                
                Should the prediction be OVER or UNDER {prediction_value}?
                Provide a brief explanation.
                """
                
                prediction = get_prediction(prompt)
                st.success(f"Prediction: {prediction}")
            else:
                st.error("Could not fetch player statistics. Please try again.")

else:  # Team page
    st.header('Team Predictions')
    
    # Team selection
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select Home Team:', team_names, key='team1_team')
    with col2:
        team2 = st.selectbox('Select Away Team:', team_names, key='team2_team')
    
    # Prediction type
    prediction_type = st.radio('Prediction Type:', 
                             ['Single Team Points', 'Combined Total Points'])
    
    # Prediction value input
    prediction_value = st.number_input('Enter prediction value:', min_value=0.0, step=0.5)
    
    # Variables selection
    st.subheader('Additional Variables')
    injuries = st.checkbox('Check for Injuries')
    rivalry = st.checkbox('Rivalry Factor')
    news = st.checkbox('Recent News/Scandals')
    weather = st.checkbox('Extreme Weather')
    
    if st.button('Predict'):
        with st.spinner('Fetching team statistics...'):
            team1_id = next(team['id'] for team in nba_teams if team['full_name'] == team1)
            team2_id = next(team['id'] for team in nba_teams if team['full_name'] == team2)
            
            team1_stats = get_team_stats(team1_id)
            team2_stats = get_team_stats(team2_id)
            
            if team1_stats is not None and team2_stats is not None:
                prompt = f"""
                Analyze the following scenario for an NBA prediction:
                Home Team: {team1}
                Away Team: {team2}
                Prediction Type: {prediction_type}
                Prediction Value: {prediction_value}
                
                Recent Stats:
                Home Team: {team1_stats.to_dict()}
                Away Team: {team2_stats.to_dict()}
                
                Consider these factors:
                - Injuries: {injuries}
                - Rivalry: {rivalry}
                - News/Scandals: {news}
                - Weather: {weather}
                
                Should the prediction be OVER or UNDER {prediction_value}?
                Provide a brief explanation.
                """
                
                prediction = get_prediction(prompt)
                st.success(f"Prediction: {prediction}")
            else:
                st.error("Could not fetch team statistics. Please try again.")