/**
 * Format timestamp safely, handling Unix timestamps (seconds) and invalid dates.
 * 
 * @param {string|number|null|undefined} timestamp - Timestamp to format
 * @returns {string} Formatted date string or '-' if missing, 'Invalid Date' if invalid
 */
export function formatTimestamp(timestamp) {
  if (!timestamp) return '-'
  
  // Handle Unix timestamp (seconds) - convert to milliseconds if needed
  let date
  if (typeof timestamp === 'string' || typeof timestamp === 'number') {
    const numTimestamp = typeof timestamp === 'string' ? parseFloat(timestamp) : timestamp
    // If timestamp is in seconds (less than year 2000 in milliseconds), multiply by 1000
    if (numTimestamp > 0 && numTimestamp < 946684800000) {
      date = new Date(numTimestamp * 1000)
    } else {
      date = new Date(numTimestamp)
    }
  } else {
    date = new Date(timestamp)
  }
  
  // Check if date is valid
  if (isNaN(date.getTime())) {
    return 'Invalid Date'
  }
  
  return date.toLocaleString()
}
