# User Signup Functionality

The AutoML Framework now supports user registration, allowing new users to create accounts and access the platform.

## Features

### Backend API

- **Signup Endpoint**: `POST /api/v1/auth/signup`
- **Input Validation**: Username, email, and password validation
- **Security**: Password hashing using bcrypt
- **Error Handling**: Comprehensive validation with clear error messages

### Frontend UI

- **Signup Form**: Integrated into the authentication page
- **Form Validation**: Client-side validation for better UX
- **Auto-login**: Automatic login after successful signup
- **Toggle**: Easy switching between login and signup modes

## API Usage

### Signup Request

```bash
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "password": "securepassword123",
    "confirm_password": "securepassword123"
  }'
```

### Successful Response

```json
{
  "id": "user_12345678",
  "username": "newuser",
  "email": "user@example.com",
  "is_active": true,
  "is_admin": false
}
```

### Error Responses

#### Password Mismatch
```json
{
  "detail": "Passwords do not match"
}
```

#### Username Already Exists
```json
{
  "detail": "Username already exists"
}
```

#### Short Password
```json
{
  "detail": "Password must be at least 6 characters long"
}
```

## Validation Rules

### Username
- Minimum 3 characters
- Only letters, numbers, and underscores allowed
- Must be unique

### Email
- Valid email format required
- Must be unique

### Password
- Minimum 6 characters
- Must match confirmation password

## UI Integration

### Using the Signup Form

1. Navigate to the authentication page
2. Click "Don't have an account? Sign up"
3. Fill in the signup form:
   - Username
   - Email address
   - Password
   - Confirm password
4. Click "Create Account"
5. Upon success, you'll be automatically logged in

### API Client Usage

```typescript
import { apiClient } from '@/lib/api-client';

const signupData = {
  username: 'newuser',
  email: 'user@example.com',
  password: 'securepassword123',
  confirm_password: 'securepassword123'
};

const result = await apiClient.signup(signupData);

if (result.error) {
  console.error('Signup failed:', result.error);
} else {
  console.log('Signup successful:', result.data);
}
```

### Auth Context Usage

```typescript
import { useAuth } from '@/contexts/AuthContext';

const { signUp } = useAuth();

const handleSignup = async () => {
  const result = await signUp({
    username: 'newuser',
    email: 'user@example.com',
    password: 'securepassword123',
    confirm_password: 'securepassword123'
  });

  if (result.error) {
    // Handle error
  } else {
    // User is now signed up and logged in
  }
};
```

## Testing

### Running Tests

```bash
# Test the signup integration
python test_signup_integration.py

# Test with the HTML test page
open test_ui_auth.html
```

### Test Coverage

- ✅ Successful user creation
- ✅ Password validation
- ✅ Username uniqueness
- ✅ Email validation
- ✅ Auto-login after signup
- ✅ CORS support
- ✅ UI integration

## Security Considerations

1. **Password Hashing**: All passwords are hashed using bcrypt before storage
2. **Input Validation**: Comprehensive server-side validation
3. **CORS**: Proper CORS headers for browser security
4. **Rate Limiting**: Built-in rate limiting for auth endpoints
5. **JWT Tokens**: Secure token-based authentication

## Demo Accounts

The system still includes demo accounts for testing:

- **Demo User**: `demo_user` / `secret`
- **Admin User**: `admin` / `secret`

## Future Enhancements

- Email verification
- Password reset functionality
- Social login integration
- User profile management
- Account deactivation/deletion

## Troubleshooting

### Common Issues

1. **"Username already exists"**: Choose a different username
2. **"Passwords do not match"**: Ensure both password fields are identical
3. **"Invalid email format"**: Use a valid email address format
4. **Connection errors**: Ensure the API server is running on port 8000

### Debug Mode

Enable debug logging to see detailed signup flow:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For issues or questions about the signup functionality, please check:

1. API server logs for backend errors
2. Browser console for frontend errors
3. Network tab for API request/response details
4. Test files for usage examples