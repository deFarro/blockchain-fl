export async function apiCall(baseUrl, endpoint, method = 'GET', body = null, apiKey) {
  const options = {
    method,
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'application/json',
    },
  }

  if (body) {
    options.body = JSON.stringify(body)
  }

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, options)
    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || error.error || 'API request failed')
    }
    return await response.json()
  } catch (error) {
    throw error
  }
}

