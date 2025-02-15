import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import calendar
import json
import os
from datetime import datetime
from pathlib import Path

def check_username():
    """Simple username authentication"""
    # If already authenticated, just return True
    if st.session_state.get('authenticated', False) and st.session_state.get('current_user'):
        return True
        
    # Show login form
    st.title("Login")
    username = st.text_input("Enter Username", key="login_username")
    
    if username:
        if username in st.secrets.get("usernames", []):
            st.session_state.authenticated = True
            st.session_state.current_user = username
            
            # Load data for this user
            filename = f"billable_hours_{username}.json"
            data_path = Path.home() / ".streamlit" / filename
            
            if data_path.exists():
                try:
                    with open(data_path, "r") as f:
                        data = json.load(f)
                        # Update session state with loaded data
                        for key, value in data.items():
                            if key == 'non_billable_weeks':
                                st.session_state[key] = set(value)
                            else:
                                st.session_state[key] = value
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.rerun()
        else:
            st.error("Invalid username")
    return False

def add_logout_button():
    """Add a logout button to the sidebar"""
    if st.sidebar.button("Logout"):
        # Clear authentication
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

def save_persistent_data():
    """Save current state to persistent storage."""
    try:
        if 'current_user' not in st.session_state:
            return
            
        data = {
            'target_hours': st.session_state.target_hours,
            'current_week': st.session_state.current_week,
            'non_billable_weeks': list(st.session_state.non_billable_weeks),
            'hours_worked': st.session_state.hours_worked,
            'daily_hours': st.session_state.daily_hours,
            'setup_complete': st.session_state.setup_complete,
            'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        data_dir = Path.home() / ".streamlit"
        data_dir.mkdir(exist_ok=True)
        
        # Use username in filename
        filename = f"billable_hours_{st.session_state.current_user}.json"
        data_path = data_dir / filename
        
        with open(data_path, "w") as f:
            json.dump(data, f)
            
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

def load_persistent_data():
    """Load data from persistent storage."""
    try:
        if 'current_user' not in st.session_state:
            return
            
        filename = f"billable_hours_{st.session_state.current_user}.json"
        data_path = Path.home() / ".streamlit" / filename
        
        if data_path.exists():
            with open(data_path, "r") as f:
                data = json.load(f)
                st.session_state.update(data)
                st.session_state.setup_complete = True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def calculate_weekly_target(total_target, non_billable_weeks, total_weeks=52):
    """Calculate weekly target hours based on available weeks."""
    available_weeks = total_weeks - len(non_billable_weeks)
    if available_weeks <= 0:
        return 0.0
    return float(total_target) / available_weeks

def generate_weeks_for_year(year):
    """Generate a list of week ranges for the year starting from Monday."""
    weeks = []
    current_date = datetime(year-1, 12, 30)  # Start from Dec 30, 2024 (Monday)
    while current_date.weekday() != 0:  # 0 represents Monday
        current_date += timedelta(days=1)
    
    while current_date.year <= year:
        week_start = current_date
        week_end = current_date + timedelta(days=4)  # Friday
        week_dates = [(week_start + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(5)]
        weeks.append({
            'week_num': len(weeks) + 1,
            'start_date': week_start.strftime('%Y-%m-%d'),
            'end_date': week_end.strftime('%Y-%m-%d'),
            'display_range': f"Week {len(weeks) + 1}: {week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}",
            'dates': week_dates
        })
        current_date += timedelta(days=7)
        if len(weeks) == 52:
            break
    return weeks

def get_current_week():
    """Determine the current week number based on today's date."""
    today = datetime.now().date()
    for week in st.session_state.weeks_data:
        start_date = datetime.strptime(week['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(week['end_date'], '%Y-%m-%d').date()
        if start_date <= today <= end_date:
            return week['week_num']
    return 1  # Default to week 1 if not found


def calculate_expected_hours(weekly_target, current_week, weeks_data):
    """Calculate expected hours based on completed weeks."""
    today = datetime.now().date()
    
    # If we're in week 6 (before Feb 16), we should count 6 full weeks
    # This is simpler than the date comparison since we know we want full weeks
    completed_weeks = current_week - 1
        
    return weekly_target * completed_weeks

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
        
    if 'weeks_data' not in st.session_state:
        st.session_state.weeks_data = generate_weeks_for_year(2025)
    
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False

    if 'target_hours' not in st.session_state:
        st.session_state.target_hours = 1600.0

    if 'current_week' not in st.session_state:
        st.session_state.current_week = get_current_week()

    if 'selected_week' not in st.session_state:
        st.session_state.selected_week = get_current_week()

    if 'non_billable_weeks' not in st.session_state:
        st.session_state.non_billable_weeks = set()
    elif not isinstance(st.session_state.non_billable_weeks, set):
        st.session_state.non_billable_weeks = set(st.session_state.non_billable_weeks)

    if 'hours_worked' not in st.session_state:
        st.session_state.hours_worked = {}
        for i in range(1, 53):
            st.session_state.hours_worked[i] = 0.0

    if 'daily_hours' not in st.session_state:
        st.session_state.daily_hours = {}
        for week in st.session_state.weeks_data:
            for date in week['dates']:
                st.session_state.daily_hours[date] = 0.0

    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Weekly"

    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

def update_weekly_hours_from_daily():
    """Update weekly hours based on daily inputs."""
    for week_num in range(1, 53):
        week_data = st.session_state.weeks_data[week_num - 1]
        total_week_hours = sum(st.session_state.daily_hours[date] for date in week_data['dates'])
        st.session_state.hours_worked[week_num] = total_week_hours

def setup_page():
    """Display the setup page for initial configuration."""
    st.title("Billable Hours Tracker Setup")
    
    # Target hours input
    target_hours = st.number_input(
        "Target Yearly Billable Hours",
        min_value=0.0,
        value=float(st.session_state.target_hours),
        step=1.0
    )
    
    # Current week selection
    current_week = st.selectbox(
        "Choose Current Week",
        options=range(1, 53),
        format_func=lambda x: st.session_state.weeks_data[x-1]['display_range']
    )
    
    # Previous hours input method
    hours_input_method = st.radio(
        "How would you like to input previously worked hours?",
        ["Total Hours", "Week by Week", "Day by Day"]
    )
    
    if hours_input_method == "Total Hours":
        total_hours = st.number_input("Total Hours Worked So Far", min_value=0.0, value=0.0, step=0.5)
        if st.button("Distribute Hours"):
            avg_hours = total_hours / current_week
            for week in range(1, current_week + 1):
                st.session_state.hours_worked[week] = float(avg_hours)
                # Distribute to daily hours
                week_data = st.session_state.weeks_data[week - 1]
                daily_avg = avg_hours / 5
                for date in week_data['dates']:
                    st.session_state.daily_hours[date] = float(daily_avg)
    
    elif hours_input_method == "Week by Week":
        st.write("Enter hours for each week up to current week:")
        for week in range(1, current_week + 1):
            week_hours = st.number_input(
                f"Hours for {st.session_state.weeks_data[week-1]['display_range']}",
                min_value=0.0,
                value=float(st.session_state.hours_worked[week]),
                step=0.5,
                key=f"week_input_{week}"
            )
            st.session_state.hours_worked[week] = float(week_hours)
    
    else:  # Day by Day
        st.write("Enter hours for each day up to current week:")
        for week in range(1, current_week + 1):
            st.subheader(st.session_state.weeks_data[week-1]['display_range'])
            cols = st.columns(5)
            for i, date in enumerate(st.session_state.weeks_data[week-1]['dates']):
                with cols[i]:
                    day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                    st.write(f"**{day_name}**")
                    st.write(date)
                    daily_hours = st.number_input(
                        "Hours",
                        min_value=0.0,
                        value=float(st.session_state.daily_hours[date]),
                        step=0.5,
                        key=f"setup_daily_{date}"
                    )
                    st.session_state.daily_hours[date] = float(daily_hours)
        update_weekly_hours_from_daily()
    
    # Non-billable weeks selection
    st.subheader("Select Non-Billable Weeks")
    cols = st.columns(4)
    for i, week in enumerate(st.session_state.weeks_data):
        col_idx = i % 4
        week_num = week['week_num']
        if cols[col_idx].checkbox(
            week['display_range'],
            value=week_num in st.session_state.non_billable_weeks,
            key=f"week_{week_num}"
        ):
            st.session_state.non_billable_weeks.add(week_num)
        else:
            st.session_state.non_billable_weeks.discard(week_num)
    
    if st.button("Save Setup"):
        st.session_state.target_hours = target_hours
        st.session_state.current_week = current_week
        st.session_state.setup_complete = True
        save_persistent_data()
        st.rerun()

def dashboard_page():
    """Display the main dashboard."""
    add_logout_button()
    st.title("Billable Hours Dashboard")
    
    # Sidebar for quick actions and week selection
    with st.sidebar:
        st.subheader("Quick Actions")
        if st.button("Return to Setup"):
            st.session_state.setup_complete = False
            st.rerun()
        
        st.subheader("Time Entry")
        # Add input mode selection
        st.session_state.input_mode = st.radio(
            "Input Mode",
            ["Weekly", "Daily"],
            key="input_mode_radio"
        )
        
        # Week selection
        st.session_state.selected_week = st.selectbox(
            "Select Week",
            options=range(1, 53),
            index=st.session_state.current_week - 1,
            format_func=lambda x: st.session_state.weeks_data[x-1]['display_range'],
            key="week_selector"
        )
        
        if st.session_state.input_mode == "Weekly":
            try:
                current_hours = st.session_state.hours_worked.get(st.session_state.selected_week, 0.0)
            except:
                current_hours = 0.0
                
            week_hours = st.number_input(
                "Hours Worked This Week",
                min_value=0.0,
                value=float(current_hours),
                step=0.5,
                key=f"weekly_hours_input_{st.session_state.selected_week}"
            )
    
            if st.button("Save Weekly Hours"):
                st.session_state.hours_worked[st.session_state.selected_week] = week_hours
                # Distribute to daily hours
                week_data = st.session_state.weeks_data[st.session_state.selected_week - 1]
                daily_avg = week_hours / 5
                for date in week_data['dates']:
                    st.session_state.daily_hours[date] = daily_avg
                save_persistent_data()
                st.success("Hours updated!")
    
            # Add Save/Load functionality
        st.markdown("---")
        st.subheader("Data Management")
        
        # Save button
        if st.button("Save Current Data"):
            success, result = save_data()
            if success:
                st.success(f"Data saved to {result}")
            else:
                st.error(f"Error saving data: {result}")
        
        # Load functionality
        uploaded_file = st.file_uploader("Load Previous Data", type=['json'])
        if uploaded_file is not None:
            success, message = load_data(uploaded_file)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    # Main content area
    if st.session_state.input_mode == "Daily":
        st.subheader(f"Daily Hours Entry for {st.session_state.weeks_data[st.session_state.selected_week-1]['display_range']}")
        
        # Create columns for each day of the week
        cols = st.columns(5)
        week_data = st.session_state.weeks_data[st.session_state.selected_week - 1]
        
        for i, date in enumerate(week_data['dates']):
            with cols[i]:
                day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                st.write(f"**{day_name}**")
                st.write(date)
                new_hours = st.number_input(
                    "Hours",
                    min_value=0.0,
                    value=st.session_state.daily_hours[date],
                    step=0.5,
                    key=f"daily_{date}"
                )
                if new_hours != st.session_state.daily_hours[date]:
                    st.session_state.daily_hours[date] = new_hours
                    update_weekly_hours_from_daily()
                    save_persistent_data()
    
    # Calculate metrics
    weekly_target = calculate_weekly_target(
    st.session_state.target_hours,
    st.session_state.non_billable_weeks
    )
    total_hours_worked = sum(st.session_state.hours_worked.values())
    # Use floor division to get completed weeks only
 
    expected_hours = calculate_expected_hours(
        weekly_target, 
        st.session_state.current_week,
        st.session_state.weeks_data
    )   
    hours_difference = total_hours_worked - expected_hours

   
    # Display metrics
# In dashboard_page(), where you display metrics
    st.subheader("Progress Metrics")
    col1, col2, col3, col4 = st.columns(4)  # Changed to 4 columns
    col1.metric("Weekly Target Hours", f"{weekly_target:.1f}")
    col2.metric("Total Hours Worked", f"{total_hours_worked:.1f}")
    col3.metric("Hours Ahead/Behind", f"{hours_difference:.1f}")
    
    # Calculate new adjusted target
    remaining_weeks = 52 - len(st.session_state.non_billable_weeks) - st.session_state.current_week
    remaining_hours = st.session_state.target_hours - total_hours_worked
    if remaining_weeks > 0:
        adjusted_weekly_target = remaining_hours / remaining_weeks
        col4.metric(
            "New Weekly Target", 
            f"{adjusted_weekly_target:.1f}",
            f"{adjusted_weekly_target - weekly_target:.1f}"  # Shows difference from original target
        )
    
    # Status message
    if abs(hours_difference) < 1:
        st.success("You are right on target! Keep it up!")
    elif hours_difference > 0:
        st.success(f"You are {hours_difference:.1f} hours ahead of pace! Keep up the great work.")
    else:
        st.error(f"You need to catch up {abs(hours_difference):.1f} hours to be on pace.")
        weeks_remaining = 52 - st.session_state.current_week
        if weeks_remaining > 0:
            extra_hours_per_week = abs(hours_difference) / weeks_remaining
            st.warning(f"Suggestion: Work an extra {extra_hours_per_week:.1f} hours per week to catch up.")
    
    # Progress graph
    st.subheader("Weekly Progress")
    hours_data = [st.session_state.hours_worked.get(i, 0.0) for i in range(1, 53)]
    df = pd.DataFrame({
        'Week': range(1, 53),
        'Hours': hours_data,
        'Target': [weekly_target if i not in st.session_state.non_billable_weeks else 0 for i in range(1, 53)]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Week'],
        y=df['Hours'],
        name='Hours Worked',
        marker_color=['red' if h < t else 'green' for h, t in zip(df['Hours'], df['Target'])]
    ))
    fig.add_trace(go.Scatter(
        x=df['Week'],
        y=df['Target'],
        name='Target',
        line=dict(color='blue', dash='dash')
    ))

    fig.update_layout(
        title='Weekly Hours vs Target',
        xaxis_title='Week Number',
        yaxis_title='Hours',
        barmode='group'
    )

    st.plotly_chart(fig)

def save_data():
    """Save current state to a JSON file."""
    data = {
        'target_hours': st.session_state.target_hours,
        'current_week': st.session_state.current_week,
        'non_billable_weeks': list(st.session_state.non_billable_weeks),
        'hours_worked': st.session_state.hours_worked,
        'daily_hours': st.session_state.daily_hours,
        'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create a filename with the current date
    filename = f'billable_hours_data_{datetime.now().strftime("%Y%m%d")}.json'
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f)
        return True, filename
    except Exception as e:
        return False, str(e)

def load_data(uploaded_file):
    """Load state from a JSON file."""
    try:
        data = json.loads(uploaded_file.getvalue().decode())
        
        st.session_state.target_hours = float(data['target_hours'])
        st.session_state.current_week = data['current_week']
        st.session_state.non_billable_weeks = set(data['non_billable_weeks'])
        st.session_state.hours_worked = {int(k): float(v) for k, v in data['hours_worked'].items()}
        st.session_state.daily_hours = {k: float(v) for k, v in data['daily_hours'].items()}
        
        return True, f"Data loaded successfully! Last saved: {data.get('last_saved', 'unknown')}"
    except Exception as e:
        return False, f"Error loading data: {str(e)}"

def main():
    initialize_session_state()
    
    if not st.session_state.get('authenticated', False):
        if check_username():
            st.rerun()
        return
    
    if not st.session_state.get('setup_complete', False):
        setup_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()

