import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import json
import os
import time
from pathlib import Path
import logging
import hashlib
import secrets

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('billable_tracker')

# Constants
TIMEOUT_SECONDS = 3600  # 1 hour
MAX_WEEKLY_HOURS = 168  # Physical maximum (24*7)
REASONABLE_WEEKLY_HOURS = 80  # Reasonable maximum for warnings
DEFAULT_TARGET_HOURS = 1600.0
DATA_DIR = Path.home() / ".streamlit"

class DataManager:
    """Class to handle all data operations with proper validation and error handling"""
    
    @staticmethod
    def get_user_data_path(username):
        """Get path to user data file"""
        if not username or not isinstance(username, str):
            raise ValueError("Invalid username")
        return DATA_DIR / f"billable_hours_{username}.json"
    
    @staticmethod
    def save_user_data(username, data):
        """Save user data with validation"""
        if not username or not data:
            logger.error("Invalid data or username for saving")
            return False, "Invalid data or username"
        
        try:
            # Convert non-serializable objects (like sets) to serializable form
            data_to_save = {
                'target_hours': float(data.get('target_hours', DEFAULT_TARGET_HOURS)),
                'current_week': int(data.get('current_week', 1)),
                'non_billable_weeks': list(data.get('non_billable_weeks', [])),
                'hours_worked': {str(k): float(v) for k, v in data.get('hours_worked', {}).items()},
                'daily_hours': {str(k): float(v) for k, v in data.get('daily_hours', {}).items()},
                'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Ensure directory exists
            data_path = DataManager.get_user_data_path(username)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(data_path, 'w') as f:
                json.dump(data_to_save, f)
            
            logger.info(f"Data saved successfully for user {username}")
            return True, "Data saved successfully"
            
        except Exception as e:
            logger.error(f"Error saving data for {username}: {str(e)}")
            return False, f"Error saving data: {str(e)}"
    
    @staticmethod
    def load_user_data(username):
        """Load user data with validation"""
        if not username:
            return False, "Invalid username", {}
            
        try:
            data_path = DataManager.get_user_data_path(username)
            
            if not data_path.exists():
                logger.info(f"No data file found for user {username}")
                return True, "No existing data found", {}
                
            with open(data_path, "r") as f:
                data = json.load(f)
                
            # Basic validation
            if not isinstance(data, dict):
                return False, "Invalid data format", {}
                
            # Convert data to expected types
            processed_data = {
                'target_hours': float(data.get('target_hours', DEFAULT_TARGET_HOURS)),
                'current_week': int(data.get('current_week', 1)),
                'non_billable_weeks': set(int(w) for w in data.get('non_billable_weeks', []) if str(w).isdigit()),
                'hours_worked': {int(k): float(v) for k, v in data.get('hours_worked', {}).items() if k.isdigit()},
                'daily_hours': {k: float(v) for k, v in data.get('daily_hours', {}).items()}
            }
            
            logger.info(f"Data loaded successfully for user {username}")
            return True, "Data loaded successfully", processed_data
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data file for user {username}")
            return False, "Data file is corrupted", {}
        except Exception as e:
            logger.error(f"Error loading data for {username}: {str(e)}")
            return False, f"Error loading data: {str(e)}", {}


class SecurityManager:
    """Handles security-related functionality"""
    
    @staticmethod
    def verify_user(username, password=None):
        """Verify user credentials"""
        try:
            valid_usernames = st.secrets.get("usernames", [])
            
            # If we're using password authentication (future enhancement)
            if password is not None and "passwords" in st.secrets:
                passwords = st.secrets.get("passwords", {})
                stored_hash = passwords.get(username)
                if not stored_hash:
                    return False
                    
                # Check password hash (this is a placeholder for future implementation)
                input_hash = SecurityManager.hash_password(password, username)
                return input_hash == stored_hash
            
            # Simple username-only authentication
            return username in valid_usernames
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    @staticmethod
    def hash_password(password, salt):
        """Hash password with salt (placeholder for future implementation)"""
        # This is a simple implementation and should be replaced with proper password hashing
        combined = password + salt
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def check_session_timeout():
        """Check if the session has timed out"""
        last_activity = st.session_state.get('last_activity', 0)
        current_time = time.time()
        
        if current_time - last_activity > TIMEOUT_SECONDS:
            logger.info("Session timeout detected")
            return True
        
        # Update last activity time
        st.session_state.last_activity = current_time
        return False


def validate_numeric_input(value, min_value=0.0, max_value=REASONABLE_WEEKLY_HOURS, default=0.0):
    """Validate numeric input with reasonable limits"""
    try:
        # Convert to float
        value = float(value)
        
        # Check if within bounds
        if value < min_value:
            logger.warning(f"Value {value} below minimum {min_value}, using minimum")
            return min_value
        elif value > max_value:
            # If over reasonable maximum but under physical maximum, warn but allow
            if value <= MAX_WEEKLY_HOURS:
                logger.warning(f"Value {value} seems high, but allowed")
                return value
            else:
                logger.error(f"Value {value} exceeds physical maximum {MAX_WEEKLY_HOURS}")
                return max_value
        
        return value
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid numeric input: {value}, using default")
        return default


def check_username():
    """Enhanced username authentication with better session management"""
    initialize_session_state()  # Make sure we initialize first
    
    # If already authenticated, check timeout and return
    if st.session_state.get('authenticated', False) and st.session_state.get('current_user'):
        if SecurityManager.check_session_timeout():
            st.session_state.authenticated = False
            st.warning("Your session has expired. Please log in again.")
            return False
        return True
        
    # Show login form
    st.title("Login")
    username = st.text_input("Enter Username", key="login_username")
    
    if username:
        if SecurityManager.verify_user(username):
            st.session_state.authenticated = True
            st.session_state.current_user = username
            st.session_state.last_activity = time.time()  # Initialize last activity time
            
            # Load user data
            success, message, data = DataManager.load_user_data(username)
            
            if success and data:
                # Update session state with loaded data
                for key, value in data.items():
                    st.session_state[key] = value
                
                # Initialize weeks data and sync hours
                st.session_state.weeks_data = generate_weeks_for_year()
                st.session_state.current_week = get_current_week()
                
                enforce_data_consistency()
                logger.info(f"User {username} logged in successfully")
            elif not success:
                st.error(f"Error loading user data: {message}")
            
            st.rerun()
        else:
            st.error("Invalid username")
            logger.warning(f"Failed login attempt for username: {username}")
    
    return False


def enforce_data_consistency():
    """Comprehensive validation and synchronization of hours data"""
    try:
        data_changed = False
        
        # 1. Initialize missing data
        if not isinstance(st.session_state.get('hours_worked'), dict):
            st.session_state.hours_worked = {i: 0.0 for i in range(1, 53)}
            data_changed = True
        else:
            # Ensure all weeks exist with proper numeric values
            for week in range(1, 53):
                if week not in st.session_state.hours_worked:
                    st.session_state.hours_worked[week] = 0.0
                    data_changed = True
                else:
                    # Convert to float to ensure consistency and validate
                    try:
                        validated_value = validate_numeric_input(
                            st.session_state.hours_worked[week]
                        )
                        if abs(validated_value - st.session_state.hours_worked[week]) > 0.01:
                            st.session_state.hours_worked[week] = validated_value
                            data_changed = True
                    except (ValueError, TypeError):
                        st.session_state.hours_worked[week] = 0.0
                        data_changed = True
        
        # 2. Initialize and validate daily hours
        if not isinstance(st.session_state.get('daily_hours'), dict):
            st.session_state.daily_hours = {}
            data_changed = True
        
        # Ensure all dates exist with proper values
        for week in st.session_state.weeks_data:
            for date_str in week['dates']:
                if date_str not in st.session_state.daily_hours:
                    st.session_state.daily_hours[date_str] = 0.0
                    data_changed = True
                else:
                    # Convert to float and validate
                    try:
                        validated_value = validate_numeric_input(
                            st.session_state.daily_hours[date_str], 
                            max_value=24.0  # Maximum hours per day
                        )
                        if abs(validated_value - st.session_state.daily_hours[date_str]) > 0.01:
                            st.session_state.daily_hours[date_str] = validated_value
                            data_changed = True
                    except (ValueError, TypeError):
                        st.session_state.daily_hours[date_str] = 0.0
                        data_changed = True
        
        # 3. Check for synchronization mode preference
        sync_mode = st.session_state.get('sync_preference', 'both')
        
        # 4. Perform bi-directional synchronization using the improved function
        for week_num in range(1, 53):
            result = update_hours_bidirectional(week_num, sync_mode)
            if result:
                data_changed = True
        
        # 5. Update data_changed flag if needed
        if data_changed:
            st.session_state.data_changed = True
            logger.info("Data consistency enforced with changes")
            
        return True, "Data consistency enforced"
    except Exception as e:
        logger.error(f"Error enforcing data consistency: {str(e)}")
        return False, f"Error enforcing data consistency: {str(e)}"


def update_hours_bidirectional(week_num=None, update_type="both"):
    """Update hours bidirectionally with synchronization options
    
    Args:
        week_num: Specific week to update, or None for all weeks
        update_type: 'weekly_to_daily', 'daily_to_weekly', or 'both'
    """
    try:
        weeks_to_update = [week_num] if week_num is not None else range(1, 53)
        
        for week in weeks_to_update:
            if week < 1 or week > 52:
                continue
                
            week_data = st.session_state.weeks_data[week - 1]
            
            # Daily to weekly (sum of daily becomes weekly total)
            if update_type in ["daily_to_weekly", "both"]:
                daily_total = sum(st.session_state.daily_hours.get(date, 0.0) for date in week_data['dates'])
                daily_total = round(daily_total, 2)  # Round to avoid float precision issues
                
                # Only update if there's a meaningful difference
                if abs(st.session_state.hours_worked.get(week, 0.0) - daily_total) > 0.01:
                    st.session_state.hours_worked[week] = daily_total
                    st.session_state.data_changed = True
                    logger.info(f"Updated weekly total for week {week} to {daily_total} (from daily)")
            
            # Weekly to daily (distribute weekly total evenly across weekdays)
            if update_type in ["weekly_to_daily", "both"]:
                weekly_total = st.session_state.hours_worked.get(week, 0.0)
                daily_avg = weekly_total / 5  # Distribute evenly across 5 days
                
                # Check if we need to update daily values
                current_daily_values = [st.session_state.daily_hours.get(date, 0.0) for date in week_data['dates']]
                current_daily_sum = sum(current_daily_values)
                
                # Only update if there's a difference to avoid unnecessary changes
                if abs(current_daily_sum - weekly_total) > 0.01:
                    # If daily values already exist but sum is different, preserve their relative proportions
                    if current_daily_sum > 0.01:  # More than 0
                        # Calculate scaling factor
                        scale_factor = weekly_total / current_daily_sum
                        # Apply scaling to each day
                        for date in week_data['dates']:
                            current_value = st.session_state.daily_hours.get(date, 0.0)
                            st.session_state.daily_hours[date] = round(current_value * scale_factor, 2)
                    else:
                        # If no meaningful daily values, distribute evenly
                        for date in week_data['dates']:
                            st.session_state.daily_hours[date] = round(daily_avg, 2)
                    
                    st.session_state.data_changed = True
                    logger.info(f"Updated daily values for week {week} to match weekly total {weekly_total}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating hours bidirectionally: {str(e)}")
        return False
        
def update_weekly_hours_from_daily(week_num=None):
    """Update weekly hours based on daily inputs - redirects to bidirectional function"""
    return update_hours_bidirectional(week_num, update_type="daily_to_weekly")


def generate_weeks_for_year(year=None):
    """Generate a list of week ranges for the year starting from Monday."""
    if year is None:
        year = datetime.now().year
    weeks = []
    
    # Start from the last Monday of previous year or first Monday of current year
    current_date = datetime(year-1, 12, 30)  # Start from Dec 30 of previous year
    
    # Move to first Monday
    while current_date.weekday() != 0:  # 0 represents Monday
        current_date += timedelta(days=1)
    
    while current_date.year <= year:
        week_start = current_date
        week_end = current_date + timedelta(days=4)  # Friday
        
        # Generate list of dates for each day in the week (Monday to Friday)
        week_dates = [(week_start + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(5)]
        
        # Create more detailed display range
        start_month = week_start.strftime('%B')
        end_month = week_end.strftime('%B')
        
        # If start and end months are the same, only show month once
        if start_month == end_month:
            display_range = f"Week {len(weeks) + 1}: {start_month} {week_start.strftime('%d')} - {week_end.strftime('%d')}, {week_end.strftime('%Y')}"
        else:
            display_range = f"Week {len(weeks) + 1}: {week_start.strftime('%B %d')} - {week_end.strftime('%B %d')}, {week_end.strftime('%Y')}"
        
        weeks.append({
            'week_num': len(weeks) + 1,
            'start_date': week_start.strftime('%Y-%m-%d'),
            'end_date': week_end.strftime('%Y-%m-%d'),
            'display_range': display_range,
            'dates': week_dates
        })
        
        current_date += timedelta(days=7)
        if len(weeks) >= 52:  # Ensure we only generate 52 weeks
            break
            
    return weeks


def get_current_week():
    """Determine the current week number based on today's date."""
    today = datetime.now().date()
    
    if not hasattr(st.session_state, 'weeks_data'):
        logger.error("weeks_data not found in session state")
        return 1
    
    for week in st.session_state.weeks_data:
        try:
            start_date = datetime.strptime(week['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(week['end_date'], '%Y-%m-%d').date()
            
            # If today is weekend (Saturday=5 or Sunday=6)
            if today.weekday() >= 5:
                # If this week's end_date was the most recent Friday, return next week
                if today > end_date and today <= end_date + timedelta(days=2):
                    return min(week['week_num'] + 1, 52)  # Don't exceed week 52
            
            # During weekdays, use normal logic
            if start_date <= today <= end_date:
                return week['week_num']
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing week data: {str(e)}")
            continue
            
    return 1  # Default to week 1 if not found


def calculate_expected_hours(weekly_target, current_week, weeks_data, non_billable_weeks):
    """Calculate expected hours based on completed weeks."""
    try:
        if current_week <= 1:
            return 0.0
            
        today = datetime.now().date()
        
        try:
            current_week_data = weeks_data[current_week - 1]  # -1 because list is 0-based
            current_week_end = datetime.strptime(current_week_data['end_date'], '%Y-%m-%d').date()
        except (IndexError, KeyError):
            logger.warning(f"Invalid current_week: {current_week}")
            return 0.0
        
        # If today is past Friday of current week, include current week
        if today > current_week_end:
            weeks_to_count = current_week
            billable_weeks_so_far = weeks_to_count - len([w for w in non_billable_weeks if w <= weeks_to_count])
        else:
            # If in middle of week, only count through previous week
            weeks_to_count = current_week - 1
            billable_weeks_so_far = weeks_to_count - len([w for w in non_billable_weeks if w <= weeks_to_count])
        
        # Calculate expected hours (with validation)
        weekly_target = validate_numeric_input(weekly_target)
        billable_weeks_so_far = max(0, billable_weeks_so_far)  # Ensure non-negative
        
        expected = weekly_target * billable_weeks_so_far
        return round(expected, 2)
        
    except Exception as e:
        logger.error(f"Error calculating expected hours: {str(e)}")
        return 0.0


def calculate_metrics_safely():
    """Calculate all metrics with proper error handling"""
    try:
        # Get target hours (with validation)
        target_hours = validate_numeric_input(
            st.session_state.get('target_hours', DEFAULT_TARGET_HOURS),
            min_value=0.0,
            max_value=10000.0  # Set a reasonable upper limit for yearly hours
        )
        
        # Ensure non_billable_weeks is a set with valid weeks
        non_billable_weeks = st.session_state.get('non_billable_weeks', set())
        if not isinstance(non_billable_weeks, set):
            non_billable_weeks = set(non_billable_weeks)
        
        # Filter to only include valid weeks (1-52)
        non_billable_weeks = {w for w in non_billable_weeks if 1 <= w <= 52}
        
        # Calculate weekly target
        available_weeks = 52 - len(non_billable_weeks)
        weekly_target = target_hours / max(1, available_weeks)  # Avoid division by zero
        
        # Safely calculate total hours worked
        hours_worked = st.session_state.get('hours_worked', {})
        total_hours = sum(validate_numeric_input(hours_worked.get(i, 0.0)) for i in range(1, 53))
        
        # Get current week (with validation)
        current_week = max(1, min(52, int(st.session_state.get('current_week', 1))))
        
        # Calculate expected hours
        expected_hours = calculate_expected_hours(
            weekly_target,
            current_week,
            st.session_state.weeks_data,
            non_billable_weeks
        )
        
        # Calculate hours difference
        hours_difference = total_hours - expected_hours
        
        # Calculate new weekly target
        remaining_weeks = 52 - current_week
        future_non_billable = len([w for w in non_billable_weeks if w > current_week])
        future_billable_weeks = max(0, remaining_weeks - future_non_billable)
        
        adjusted_target = 0.0
        if future_billable_weeks > 0:
            remaining_hours = max(0, target_hours - total_hours)
            adjusted_target = remaining_hours / future_billable_weeks
        
        return {
            'weekly_target': round(weekly_target, 2),
            'total_hours': round(total_hours, 2),
            'hours_difference': round(hours_difference, 2),
            'adjusted_target': round(adjusted_target, 2),
            'expected_hours': round(expected_hours, 2),
            'remaining_billable_weeks': future_billable_weeks
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation error: {str(e)}")
        return {
            'weekly_target': 0.0,
            'total_hours': 0.0,
            'hours_difference': 0.0,
            'adjusted_target': 0.0,
            'expected_hours': 0.0,
            'remaining_billable_weeks': 0
        }


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
        
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = time.time()

    # Year and week data
    if 'current_year' not in st.session_state:
        st.session_state.current_year = datetime.now().year

    if 'weeks_data' not in st.session_state:
        st.session_state.weeks_data = generate_weeks_for_year(st.session_state.current_year)
    
    # Setup and navigation state
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False

    if 'page' not in st.session_state:
        st.session_state.page = 'main'

    # Hours tracking data
    if 'target_hours' not in st.session_state:
        st.session_state.target_hours = DEFAULT_TARGET_HOURS

    if 'current_week' not in st.session_state:
        st.session_state.current_week = get_current_week()

    if 'selected_week' not in st.session_state:
        st.session_state.selected_week = get_current_week()

    # Non-billable weeks
    if 'non_billable_weeks' not in st.session_state:
        st.session_state.non_billable_weeks = set()
    elif not isinstance(st.session_state.non_billable_weeks, set):
        st.session_state.non_billable_weeks = set(st.session_state.non_billable_weeks)

    # Hours data structures
    if 'hours_worked' not in st.session_state:
        st.session_state.hours_worked = {i: 0.0 for i in range(1, 53)}
    
    if 'daily_hours' not in st.session_state:
        st.session_state.daily_hours = {}
        # Initialize all dates to zero
        for week in st.session_state.weeks_data:
            for date_str in week['dates']:
                st.session_state.daily_hours[date_str] = 0.0

    # UI state
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Weekly"

    # Data management state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'data_changed' not in st.session_state:
        st.session_state.data_changed = False
        
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = 0


def auto_save():
    """Auto-save with debounce mechanism and error handling"""
    try:
        if st.session_state.get('data_changed', False) and st.session_state.get('current_user'):
            # Check if last save was more than 2 seconds ago
            current_time = time.time()
            last_save_time = st.session_state.get('last_save_time', 0)
            
            if current_time - last_save_time > 2:
                # Collect data to save
                data = {
                    'target_hours': st.session_state.get('target_hours', DEFAULT_TARGET_HOURS),
                    'current_week': st.session_state.get('current_week', 1),
                    'non_billable_weeks': st.session_state.get('non_billable_weeks', set()),
                    'hours_worked': st.session_state.get('hours_worked', {}),
                    'daily_hours': st.session_state.get('daily_hours', {})
                }
                
                # Save data
                success, message = DataManager.save_user_data(st.session_state.current_user, data)
                
                if success:
                    st.session_state.data_changed = False
                    st.session_state.last_save_time = current_time
                    logger.info("Auto-save completed successfully")
                else:
                    logger.warning(f"Auto-save failed: {message}")
    except Exception as e:
        logger.error(f"Auto-save error: {str(e)}")


def dashboard_page():
    """Display the main dashboard with improved error handling and UI."""
    # Display logout button
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    
    st.title("Billable Hours Dashboard")
    
    # Ensure data consistency
    enforce_data_consistency()
    
    # Display progress bar
    try:
        yearly_target = float(st.session_state.get('target_hours', DEFAULT_TARGET_HOURS))
        total_hours = sum(float(st.session_state.hours_worked.get(i, 0.0)) for i in range(1, 53))
        percentage = min(100, (total_hours / yearly_target * 100)) if yearly_target > 0 else 0
        
        st.markdown(f"### Overall Progress: {percentage:.1f}% Complete")
        st.progress(percentage / 100)
        st.markdown(f"**{total_hours:.1f}** hours of **{yearly_target:.1f}** target hours")
    except Exception as e:
        logger.error(f"Error displaying progress: {str(e)}")
        st.warning("Unable to display progress. Please check your data.")
    
    # Sidebar for quick actions and week selection
    with st.sidebar:
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Setup", key="sidebar_return_setup"):
                st.session_state.setup_complete = False
                st.rerun()
        
        with col2:
            if st.button("Settings", key="sidebar_settings"):
                st.session_state.page = "data_management"
                st.rerun()

        st.subheader("Time Entry")
        
        # Add input mode selection
        st.session_state.input_mode = st.radio(
            "Input Mode",
            ["Weekly", "Daily"],
            key="input_mode_radio"
        )
        
        # Sync data if input mode changed
        if 'previous_input_mode' not in st.session_state or st.session_state.previous_input_mode != st.session_state.input_mode:
            update_weekly_hours_from_daily()
        st.session_state.previous_input_mode = st.session_state.input_mode
        
        # Week selection with better formatting
        try:
            current_week_index = max(0, min(51, st.session_state.current_week - 1))
            st.session_state.selected_week = st.selectbox(
                "Select Week",
                options=range(1, 53),
                index=current_week_index,
                format_func=lambda x: st.session_state.weeks_data[x-1]['display_range'],
                key="week_selector"
            )
        except Exception as e:
            logger.error(f"Error with week selection: {str(e)}")
            st.session_state.selected_week = 1
        
        # Weekly hours input in sidebar
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
                key=f"sidebar_weekly_hours_input_{st.session_state.selected_week}"
            )

            if week_hours != current_hours:
                # Validate input
                validated_hours = validate_numeric_input(week_hours)
                if validated_hours > REASONABLE_WEEKLY_HOURS:
                    st.warning(f"Warning: {validated_hours} hours seems high for one week")
                
                # Update weekly total
                st.session_state.hours_worked[st.session_state.selected_week] = validated_hours
                
                # Automatically sync to daily hours with smart distribution
                update_hours_bidirectional(st.session_state.selected_week, "weekly_to_daily")
                
                st.session_state.data_changed = True
                auto_save()
                
                # Show feedback about what happened
                st.info(f"Updated weekly hours and automatically distributed to daily entries")

            sync_mode = st.radio(
                "Distribution Method", 
                ["Automatic", "Manual"], 
                index=0,
                help="Automatic syncs daily and weekly totals. Manual gives you more control."
            )
            
            if sync_mode == "Manual" and st.button("Distribute Evenly", key=f"distribute_weekly_{st.session_state.selected_week}"):
                # Get validated hours
                validated_hours = validate_numeric_input(week_hours)
                st.session_state.hours_worked[st.session_state.selected_week] = validated_hours
                
                # Distribute to daily hours
                week_data = st.session_state.weeks_data[st.session_state.selected_week - 1]
                daily_avg = validated_hours / 5
                for date in week_data['dates']:
                    st.session_state.daily_hours[date] = daily_avg
                
                st.session_state.data_changed = True
                auto_save()
                st.success("✅ Hours distributed evenly across weekdays")
                    
        # Daily input mode - improved with validation
        else:
            try:
                week_data = st.session_state.weeks_data[st.session_state.selected_week - 1]
                for date in week_data['dates']:
                    day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                    st.write(f"**{day_name}** ({date})")
                    current_value = st.session_state.daily_hours.get(date, 0.0)
                    new_hours = st.number_input(
                        "Hours",
                        min_value=0.0,
                        max_value=24.0,  # Maximum hours per day
                        value=float(current_value),
                        step=0.5,
                        key=f"daily_{date}"
                    )
                    
                    if new_hours != current_value:
                        # Validate hours
                        validated_hours = validate_numeric_input(new_hours, max_value=24.0)
                        
                        # Show warning if hours seem high
                        if validated_hours > 12.0:
                            st.warning(f"Warning: {validated_hours} hours is a long workday")
                            
                        # Update daily hours
                        st.session_state.daily_hours[date] = validated_hours
                        
                        # Find which week this date belongs to
                        for i, week_data in enumerate(st.session_state.weeks_data):
                            if date in week_data['dates']:
                                week_num = i + 1
                                break
                        
                        # Update corresponding weekly total
                        update_hours_bidirectional(week_num, "daily_to_weekly")
                        
                        st.session_state.data_changed = True
                        auto_save()
            except Exception as e:
                logger.error(f"Error in daily hours input: {str(e)}")
                st.warning("Error displaying daily hours input")

    # Main content area - Quick Hours Update
    st.subheader("Quick Hours Update")
    col1, col2, col3 = st.columns([2, 1, 1.5])
    
    with col1:
        col1.markdown("##### Current Week")
        try:
            current_input_week = st.selectbox(
                "",
                options=range(1, 53),
                index=st.session_state.current_week - 1,
                format_func=lambda x: st.session_state.weeks_data[x-1]['display_range'],
                help="Which week are you updating through?",
                key="quick_update_week_select",
                label_visibility="collapsed"
            )
        except Exception as e:
            logger.error(f"Error with quick update week selection: {str(e)}")
            current_input_week = 1
            
        st.markdown("")  # Add some spacing
        
    with col2:
        # Calculate current total with error handling
        try:
            current_total = sum(float(st.session_state.hours_worked.get(i, 0.0)) for i in range(1, current_input_week + 1))
        except Exception as e:
            logger.error(f"Error calculating current total: {str(e)}")
            current_total = 0.0
            
        total_hours = st.number_input(
            "Total Hours So Far",
            min_value=0.0,
            value=current_total,
            step=0.5
        )
    
    # Calculate billable weeks up to current_input_week with error handling
    try:
        billable_weeks = current_input_week - len([w for w in st.session_state.non_billable_weeks if w <= current_input_week])
    except Exception as e:
        logger.error(f"Error calculating billable weeks: {str(e)}")
        billable_weeks = 0
    
    with col3:
        if billable_weeks > 0:
            if st.button("Distribute Hours", key="distribute_hours_initial"):
                try:
                    # Reset all hours to 0
                    for week in range(1, 53):
                        st.session_state.hours_worked[week] = 0.0
                        week_data = st.session_state.weeks_data[week - 1]
                        for date in week_data['dates']:
                            st.session_state.daily_hours[date] = 0.0
                    
                    # Validate input hours
                    validated_total = validate_numeric_input(total_hours, max_value=10000.0)
                    
                    # Calculate average hours for billable weeks
                    avg_hours = validated_total / billable_weeks
                    
                    # Distribute hours only to billable weeks up to current_input_week
                    hours_distributed = 0
                    for week in range(1, current_input_week + 1):
                        if week not in st.session_state.non_billable_weeks:
                            st.session_state.hours_worked[week] = float(avg_hours)
                            week_data = st.session_state.weeks_data[week - 1]
                            daily_avg = avg_hours / 5
                            for date in week_data['dates']:
                                st.session_state.daily_hours[date] = float(daily_avg)
                            hours_distributed += avg_hours
                            
                    # Sync and save the changes
                    update_weekly_hours_from_daily()
                    st.session_state.data_changed = True
                    auto_save()
                    
                    # Show success message
                    st.success(f"Distributed {hours_distributed:.1f} hours across {billable_weeks} billable weeks ({avg_hours:.1f} hours/week)")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error distributing hours: {str(e)}")
                    st.error("Error distributing hours. Please try again.")
        
            # Add explanation text below button
            st.caption("* This will evenly distribute your total hours across all billable weeks up to the selected week")
        else:
            st.warning(f"No billable weeks found up to week {current_input_week}. Please check vacation weeks selection.")

    # Calculate and display metrics with improved error handling
    try:
        # Sync data
        update_weekly_hours_from_daily()
        metrics = calculate_metrics_safely()
        
        # Display metrics
        st.subheader("Progress Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Weekly Target",
            f"{metrics['weekly_target']:.1f}"
        )
        
        col2.metric(
            "Total Hours",
            f"{metrics['total_hours']:.1f}"
        )
        
        col3.metric(
            "Hours +/-",
            f"{metrics['hours_difference']:.1f}",
            delta=f"{metrics['hours_difference']:.1f}",
            delta_color="normal"
        )
        
        # Only show delta if we have an adjusted target
        if metrics['adjusted_target'] > 0:
            delta = float(metrics['adjusted_target']) - float(metrics['weekly_target'])
            col4.metric(
                "New Target",
                f"{metrics['adjusted_target']:.1f}",
                f"{delta:.1f}"
            )
        else:
            col4.metric(
                "New Target",
                "N/A"
            )
            
        # Status messages
        if abs(metrics['hours_difference']) < 1:
            st.success("✅ You are right on target! Keep it up!")
        elif metrics['hours_difference'] > 0:
            st.success(f"✅ You are {metrics['hours_difference']:.1f} hours ahead of pace!")
        else:
            st.warning(f"⚠️ You need to catch up {abs(metrics['hours_difference']):.1f} hours to be on pace.")

    
        # Weekly progress visualization
        st.subheader("Weekly Progress")
        try:
            hours_data = [st.session_state.hours_worked.get(i, 0.0) for i in range(1, 53)]
            df = pd.DataFrame({
                'Week': range(1, 53),
                'Hours': hours_data,
                'Target': [metrics['weekly_target'] if i not in st.session_state.non_billable_weeks else 0 for i in range(1, 53)],
                'Status': ['Non-Billable' if i in st.session_state.non_billable_weeks else 
                          ('Complete' if i < st.session_state.current_week else 
                           ('Current' if i == st.session_state.current_week else 'Future')) 
                          for i in range(1, 53)]
            })

            # Create a more informative chart
            fig = go.Figure()
            
            # Add bars for hours worked, colored by status
            color_map = {
                'Complete': 'green',
                'Current': 'blue',
                'Future': 'lightgray',
                'Non-Billable': 'darkgray'
            }
            
            # Add bars with appropriate colors
            for status in ['Complete', 'Current', 'Future', 'Non-Billable']:
                status_df = df[df['Status'] == status]
                fig.add_trace(go.Bar(
                    x=status_df['Week'],
                    y=status_df['Hours'],
                    name=status,
                    marker_color=color_map[status]
                ))
            
            # Add target line
            fig.add_trace(go.Scatter(
                x=df['Week'],
                y=df['Target'],
                name='Weekly Target',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                title='Weekly Hours vs Target',
                xaxis_title='Week Number',
                yaxis_title='Hours',
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error creating weekly progress chart: {str(e)}")
            st.error("Unable to display weekly progress chart")

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error("Error calculating metrics. Please check your data.")


def data_management_page():
    """Settings & Data Management page with improved error handling"""
    # Ensure data is synchronized
    update_weekly_hours_from_daily()
    
    st.title("Settings & Data Management")
    
    # Navigation
    if st.button("← Return to Dashboard"):
        st.session_state.page = 'main'
        st.rerun()

    # Create tabs for different data categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Basic Settings", 
        "Weekly Hours", 
        "Daily Hours",
        "Non-Billable Weeks",
        "Data Tools"
    ])

    with tab1:
        st.header("Basic Settings")
        
        # Target Hours with current value display
        st.subheader("Target Hours")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"Current: {st.session_state.target_hours}")
        with col2:
            try:
                new_target = st.number_input(
                    "Update Target Hours",
                    min_value=0.0,
                    max_value=10000.0,
                    value=float(st.session_state.target_hours),
                    step=10.0
                )
                if new_target != st.session_state.target_hours:
                    st.session_state.target_hours = new_target
                    st.session_state.data_changed = True
                    auto_save()
                    st.success("Target hours updated!")
            except Exception as e:
                logger.error(f"Error updating target hours: {str(e)}")
                st.error("Error updating target hours")

        # Current Week Display/Edit
        st.subheader("Current Week")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"Current: Week {st.session_state.current_week}")
        with col2:
            try:
                new_week = st.number_input(
                    "Update Current Week",
                    min_value=1,
                    max_value=52,
                    value=int(st.session_state.current_week)
                )
                if new_week != st.session_state.current_week:
                    st.session_state.current_week = new_week
                    st.session_state.data_changed = True
                    auto_save()
                    st.success("Current week updated!")
            except Exception as e:
                logger.error(f"Error updating current week: {str(e)}")
                st.error("Error updating current week")

    with tab2:
        st.header("Weekly Hours Data")
        
        try:
            # Add search/filter option
            week_filter = st.text_input("Filter weeks (enter week number)", "")
            
            # Convert weeks data to dataframe for easier handling
            weeks_df = pd.DataFrame([
                {
                    'week_num': i,
                    'display': st.session_state.weeks_data[i-1]['display_range'],
                    'hours': float(st.session_state.hours_worked.get(i, 0.0)),
                    'is_billable': i not in st.session_state.non_billable_weeks
                } for i in range(1, 53)
            ])
            
            # Apply filter if provided
            if week_filter:
                try:
                    filter_num = int(week_filter)
                    weeks_df = weeks_df[weeks_df['week_num'] == filter_num]
                except ValueError:
                    # If not a number, filter by text
                    weeks_df = weeks_df[weeks_df['display'].str.contains(week_filter, case=False)]
            
            # Determine grid layout
            cols_per_row = 3
            
            # Split into chunks for rows
            week_chunks = [weeks_df.iloc[i:i+cols_per_row] for i in range(0, len(weeks_df), cols_per_row)]
            
            # Create the grid
            for chunk in week_chunks:
                cols = st.columns(cols_per_row)
                
                for i, (_, row) in enumerate(chunk.iterrows()):
                    week_num = int(row['week_num'])
                    with cols[i]:
                        # Add visual indicator for non-billable weeks
                        if not row['is_billable']:
                            st.markdown(f"**Week {week_num} (Non-Billable)**")
                        else:
                            st.markdown(f"**Week {week_num}**")
                        
                        # Show the date range
                        st.caption(row['display'])
                        
                        # Week hours input with proper validation
                        try:
                            current_value = float(st.session_state.hours_worked.get(week_num, 0.0))
                            new_hours = st.number_input(
                                "Hours",
                                min_value=0.0,
                                max_value=MAX_WEEKLY_HOURS,
                                value=current_value,
                                step=0.5,
                                key=f"manage_week_{week_num}",
                                label_visibility="collapsed"
                            )
                            
                            # Handle value change
                            if abs(new_hours - current_value) > 0.01:
                                # Validate hours
                                validated_hours = validate_numeric_input(new_hours)
                                st.session_state.hours_worked[week_num] = validated_hours
                                
                                # Update daily hours
                                update_hours_bidirectional(week_num, st.session_state.sync_preference)
                                
                                st.session_state.data_changed = True
                                auto_save()
                        except Exception as e:
                            st.error(f"Error with week {week_num}: {str(e)}")
            
            # If no weeks match filter
            if len(weeks_df) == 0:
                st.warning("No weeks match your filter criteria")
                
        except Exception as e:
            logger.error(f"Error displaying weekly hours grid: {str(e)}")
            st.error(f"Error displaying weekly hours data: {str(e)}")
            
            # Fallback basic display in case of error
            st.warning("Displaying simplified view due to error")
            for week_num in range(1, 53):
                try:
                    st.text(f"Week {week_num}: {st.session_state.hours_worked.get(week_num, 0.0)} hours")
                except:
                    pass

    with tab3:
        st.header("Daily Hours Data")
        
        try:
            # Week selector for daily data with error handling
            week_options = list(range(1, 53))
            selected_week = st.selectbox(
                "Select Week to View/Edit Daily Hours",
                options=week_options,
                format_func=lambda x: st.session_state.weeks_data[x-1]['display_range'] if 1 <= x <= 52 
                              and len(st.session_state.weeks_data) >= x else f"Week {x}",
                key="daily_week_selector"
            )
            
            # Retrieve week data safely
            if 1 <= selected_week <= 52 and len(st.session_state.weeks_data) >= selected_week:
                week_data = st.session_state.weeks_data[selected_week - 1]
                
                # Display status for the selected week
                if selected_week in st.session_state.non_billable_weeks:
                    st.info("This is marked as a non-billable week")
                
                # Show summary of total hours for this week
                week_total = sum(st.session_state.daily_hours.get(date, 0.0) for date in week_data['dates'])
                st.metric("Week Total", f"{week_total:.1f} hours")
                
                # Create a tabular layout for daily hours
                daily_data = []
                for date in week_data['dates']:
                    try:
                        day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                        day_date = datetime.strptime(date, '%Y-%m-%d').strftime('%b %d, %Y')
                        hours = float(st.session_state.daily_hours.get(date, 0.0))
                        daily_data.append({
                            'day': day_name,
                            'date': day_date,
                            'full_date': date,
                            'hours': hours
                        })
                    except Exception as e:
                        logger.error(f"Error processing date {date}: {str(e)}")
                        continue
                
                # Create a dataframe for display
                df = pd.DataFrame(daily_data)
                
                # Display each day in its own row for better organization
                for _, row in df.iterrows():
                    with st.container():
                        cols = st.columns([2, 3])
                        with cols[0]:
                            st.markdown(f"**{row['day']}**")
                            st.caption(row['date'])
                        
                        with cols[1]:
                            try:
                                new_hours = st.number_input(
                                    "Hours",
                                    min_value=0.0,
                                    max_value=24.0,
                                    value=float(row['hours']),
                                    step=0.5,
                                    key=f"daily_edit_{row['full_date']}"
                                )
                                
                                if abs(new_hours - row['hours']) > 0.01:
                                    # Validate hours
                                    validated_hours = validate_numeric_input(new_hours, max_value=24.0)
                                    
                                    # Show warning if hours seem high
                                    if validated_hours > 12.0:
                                        st.warning(f"{validated_hours} hours is a long workday")
                                    
                                    # Update daily hours
                                    st.session_state.daily_hours[row['full_date']] = validated_hours
                                    
                                    # Update corresponding weekly total
                                    update_hours_bidirectional(selected_week, st.session_state.sync_preference)
                                    
                                    st.session_state.data_changed = True
                                    auto_save()
                            except Exception as e:
                                st.error(f"Error with {row['day']} hours: {str(e)}")
                
                # Add quick distribution options
                st.subheader("Quick Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Distribute Weekly Total Evenly", key=f"distribute_evenly_{selected_week}"):
                        week_total = st.session_state.hours_worked.get(selected_week, 0.0)
                        daily_avg = week_total / 5
                        
                        # Update all days in this week
                        for date in week_data['dates']:
                            st.session_state.daily_hours[date] = daily_avg
                            
                        st.session_state.data_changed = True
                        auto_save()
                        st.success(f"Distributed {week_total:.1f} hours evenly ({daily_avg:.1f}/day)")
                        st.rerun()
                
                with col2:
                    if st.button("Clear This Week", key=f"clear_week_{selected_week}"):
                        # Clear all days in this week
                        for date in week_data['dates']:
                            st.session_state.daily_hours[date] = 0.0
                            
                        # Update weekly total
                        st.session_state.hours_worked[selected_week] = 0.0
                        
                        st.session_state.data_changed = True
                        auto_save()
                        st.success("Week cleared")
                        st.rerun()
            else:
                st.error(f"Invalid week selection: {selected_week}")
                
        except Exception as e:
            logger.error(f"Error in daily hours tab: {str(e)}")
            st.error(f"Error displaying daily hours: {str(e)}")
            
            # Fallback simple display
            st.warning("Displaying simplified view due to error")
            try:
                if 1 <= selected_week <= 52 and len(st.session_state.weeks_data) >= selected_week:
                    week_data = st.session_state.weeks_data[selected_week - 1]
                    for date in week_data['dates']:
                        hours = st.session_state.daily_hours.get(date, 0.0)
                        st.text(f"{date}: {hours} hours")
            except:
                st.error("Unable to display any data. Please try another week.")

    with tab4:
        st.header("Non-Billable Weeks")
        
        try:
            # Display current non-billable weeks
            st.subheader("Current Non-Billable Weeks")
            if st.session_state.non_billable_weeks:
                st.info(f"Non-billable weeks: {sorted(list(st.session_state.non_billable_weeks))}")
            else:
                st.info("No non-billable weeks set")
                
            # Week selection for non-billable weeks with improved error handling
            try:
                selected_weeks = st.multiselect(
                    "Select Non-Billable Weeks",
                    options=[week['display_range'] for week in st.session_state.weeks_data],
                    default=[st.session_state.weeks_data[i-1]['display_range'] for i in st.session_state.non_billable_weeks]
                )
                
                # Update non-billable weeks based on selection
                new_non_billable = set(
                    i+1 for i, week in enumerate(st.session_state.weeks_data) 
                    if week['display_range'] in selected_weeks
                )
                
                if new_non_billable != st.session_state.non_billable_weeks:
                    st.session_state.non_billable_weeks = new_non_billable
                    st.session_state.data_changed = True
                    st.success("Non-billable weeks updated!")
                    auto_save()
            except Exception as e:
                logger.error(f"Error updating non-billable weeks: {str(e)}")
                st.error("Error updating non-billable weeks")
        except Exception as e:
            logger.error(f"Error in non-billable weeks tab: {str(e)}")
            st.error("Error displaying non-billable weeks information")

    with tab5:
        st.header("Data Tools")
        
        # Data Reset Section
        st.subheader("Reset Data")
        with st.expander("Clear All Data"):
            st.warning("⚠️ This will reset all hours to zero!")
            if st.button("Clear All Data", type="primary"):
                confirm = st.checkbox("Are you sure? This will reset everything to zero.")
                if confirm:
                    try:
                        # Initialize hours_worked with zeros
                        st.session_state.hours_worked = {i: 0.0 for i in range(1, 53)}
                        
                        # Initialize daily_hours with zeros
                        st.session_state.daily_hours = {}
                        for week in st.session_state.weeks_data:
                            for date in week['dates']:
                                st.session_state.daily_hours[date] = 0.0

                        # Save the cleared data
                        st.session_state.data_changed = True
                        auto_save()
                        st.success("✅ All data has been cleared!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error while clearing data: {str(e)}")
                        st.error(f"Error while clearing data: {str(e)}")
        
        # Data Export/Import
        st.subheader("Export/Import Data")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Export Data"):
                try:
                    # Prepare data for export
                    export_data = {
                        'target_hours': st.session_state.target_hours,
                        'current_week': st.session_state.current_week,
                        'non_billable_weeks': list(st.session_state.non_billable_weeks),
                        'hours_worked': st.session_state.hours_worked,
                        'daily_hours': st.session_state.daily_hours,
                        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Convert to JSON
                    export_json = json.dumps(
                        {str(k): v for k, v in export_data['hours_worked'].items()},
                        indent=2
                    )
                    
                    st.download_button(
                        label="Download Hours Data",
                        data=export_json,
                        file_name=f"billable_hours_export_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    logger.error(f"Error exporting data: {str(e)}")
                    st.error("Error preparing data for export")
        
        with col2:
            with st.expander("Import Data"):
                st.warning("⚠️ Importing will overwrite your current data")
                uploaded_file = st.file_uploader("Upload Hours Data JSON", type="json")
                
                if uploaded_file is not None:
                    try:
                        import_data = json.load(uploaded_file)
                        
                        # Basic validation
                        if not isinstance(import_data, dict):
                            st.error("Invalid JSON format - must be an object")
                        else:
                            # Convert keys to integers for hours_worked
                            st.session_state.hours_worked = {
                                int(k): float(v) for k, v in import_data.items() if k.isdigit()
                            }
                            
                            # Update weekly data from imported data
                            update_weekly_hours_from_daily()
                            
                            st.session_state.data_changed = True
                            auto_save()
                            st.success("✅ Data imported successfully!")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON file")
                    except Exception as e:
                        logger.error(f"Error importing data: {str(e)}")
                        st.error(f"Error importing data: {str(e)}")
        
        # Debug Information
        st.subheader("Debug Information")
        with st.expander("View Current Data State"):
            try:
                st.json({
                    'target_hours': st.session_state.target_hours,
                    'current_week': st.session_state.current_week,
                    'non_billable_weeks': list(st.session_state.non_billable_weeks),
                    'hours_worked': {
                        str(k): v for k, v in st.session_state.hours_worked.items()
                    },
                    'daily_hours': {
                        k: v for k, v in list(st.session_state.daily_hours.items())[:10]  # Show just first 10 for brevity
                    }
                })
            except Exception as e:
                logger.error(f"Error displaying debug info: {str(e)}")
                st.error("Error displaying debug information")


def setup_page():
    """Display the simplified setup page with better UI and error handling."""
    st.title("Billable Hours Tracker Setup")
    
    # Add data changed tracker
    if 'data_changed' not in st.session_state:
        st.session_state.data_changed = False
    
    # Target hours input with better validation
    try:
        target_hours = st.number_input(
            "Target Yearly Billable Hours",
            min_value=0.0,
            max_value=10000.0,  # Reasonable upper limit
            value=float(st.session_state.target_hours),
            step=10.0,
            help="Your yearly billable hours target (e.g. 1600, 2000)"
        )
        
        if target_hours != st.session_state.target_hours:
            st.session_state.target_hours = target_hours  # Update immediately
            st.session_state.data_changed = True
            auto_save()  # Save changes immediately
    except Exception as e:
        logger.error(f"Error updating target hours: {str(e)}")
        st.error("Error updating target hours")
        target_hours = st.session_state.target_hours  # Fallback to current value
    
    # Improved UI for non-billable weeks selection
    st.subheader("Select Non-Billable Weeks")
    st.caption("Choose weeks for holidays, vacation, or other non-billable time")
    
    try:
        # More organized display with month grouping
        current_month = None
        cols_per_row = 4
        col_idx = 0
        cols = st.columns(cols_per_row)
        
        # Keep track of changes
        old_non_billable = set(st.session_state.non_billable_weeks)
        
        for i, week in enumerate(st.session_state.weeks_data):
            week_start = datetime.strptime(week['start_date'], '%Y-%m-%d')
            month = week_start.strftime("%B")
            
            # Start a new row for each month
            if month != current_month:
                st.markdown(f"##### {month}")
                current_month = month
                cols = st.columns(cols_per_row)
                col_idx = 0
            
            week_num = week['week_num']
            week_display = f"Week {week_num}: {week_start.strftime('%d')}-{(week_start + timedelta(days=4)).strftime('%d')}"
            was_selected = week_num in old_non_billable
            
            with cols[col_idx % cols_per_row]:
                if cols[col_idx % cols_per_row].checkbox(
                    week_display,
                    value=was_selected,
                    key=f"week_{week_num}"
                ):
                    if not was_selected:
                        st.session_state.non_billable_weeks.add(week_num)
                        st.session_state.data_changed = True
                else:
                    if was_selected:
                        st.session_state.non_billable_weeks.discard(week_num)
                        st.session_state.data_changed = True
            
            col_idx += 1
            
            # If there were changes, save immediately
            if st.session_state.non_billable_weeks != old_non_billable:
                auto_save()
                # Update the old set to track new changes
                old_non_billable = set(st.session_state.non_billable_weeks)
        
        # Show current non-billable weeks for verification
        if st.session_state.non_billable_weeks:
            st.info(f"Non-billable weeks selected: {len(st.session_state.non_billable_weeks)}")
            
            # Weekly target with current selections
            available_weeks = 52 - len(st.session_state.non_billable_weeks)
            if available_weeks > 0:
                weekly_target = target_hours / available_weeks
                st.success(f"Your weekly target will be: {weekly_target:.1f} hours")
            else:
                st.error("You've selected all weeks as non-billable. Please unselect some weeks.")
    except Exception as e:
        logger.error(f"Error in non-billable weeks selection: {str(e)}")
        st.error("Error displaying non-billable weeks selector")
    
    # Save button with enhanced error handling
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Save Setup", type="primary"):
            try:
                if 52 - len(st.session_state.non_billable_weeks) <= 0:
                    st.error("You must have at least one billable week")
                else:
                    st.session_state.target_hours = target_hours
                    st.session_state.setup_complete = True
                    st.session_state.data_changed = True
                    
                    # Save data with proper error handling
                    data = {
                        'target_hours': st.session_state.target_hours,
                        'current_week': st.session_state.current_week,
                        'non_billable_weeks': st.session_state.non_billable_weeks,
                        'hours_worked': st.session_state.hours_worked,
                        'daily_hours': st.session_state.daily_hours
                    }
                    
                    success, message = DataManager.save_user_data(st.session_state.current_user, data)
                    
                    if success:
                        st.session_state.data_changed = False
                        st.success("✅ Setup saved successfully!")
                        # Give user time to see success message
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Error saving setup: {message}")
                        
            except Exception as e:
                logger.error(f"Error during setup save: {str(e)}")
                st.error(f"Error during setup: {str(e)}")


def main():
    """Main application with improved error handling and security"""
    try:
        # Initialize session state 
        initialize_session_state()
        
        # Set default sync preference if not already set
        if 'sync_preference' not in st.session_state:
            st.session_state.sync_preference = 'both'
        
        # Check authentication
        if not check_username():
            return
        
        # Refresh session last activity time
        st.session_state.last_activity = time.time()
        
        # Check for session timeout during the session
        if SecurityManager.check_session_timeout():
            st.warning("Your session has expired. Please log in again.")
            st.session_state.authenticated = False
            st.rerun()
            return
            
        # Add a settings section in the sidebar if logged in
        if st.session_state.authenticated:
            with st.sidebar:
                with st.expander("Advanced Settings"):
                    st.session_state.sync_preference = st.radio(
                        "Data Sync Mode",
                        options=["daily_to_weekly", "weekly_to_daily", "both"],
                        index=["daily_to_weekly", "weekly_to_daily", "both"].index(
                            st.session_state.get("sync_preference", "both")
                        ),
                        format_func=lambda x: {
                            "daily_to_weekly": "Daily → Weekly (daily entries control totals)",
                            "weekly_to_daily": "Weekly → Daily (weekly totals distribute to days)",
                            "both": "Bidirectional (smart sync in both directions)"
                        }[x],
                        help="Choose how hours data should synchronize between daily and weekly views"
                    )
        
        # Ensure data consistency
        enforce_data_consistency()
        auto_save()
        
        # Display appropriate page
        if not st.session_state.setup_complete:
            setup_page()
        elif st.session_state.page == "data_management":
            data_management_page()
        else:
            dashboard_page()
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred. Please refresh the page and try again.")
        

if __name__ == "__main__":
    main()
