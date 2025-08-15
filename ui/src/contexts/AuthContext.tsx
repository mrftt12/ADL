import { createContext, useContext, useEffect, useState } from 'react';
import { apiClient } from '@/lib/api-client';

interface User {
  id: string;
  username: string;
  email?: string;
  is_active: boolean;
  is_admin: boolean;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: (username: string, password: string) => Promise<{ error?: string }>;
  signUp: (userData: {
    username: string;
    email: string;
    password: string;
    confirm_password: string;
  }) => Promise<{ error?: string }>;
  signOut: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated on app start
    const checkAuth = async () => {
      if (apiClient.isAuthenticated()) {
        try {
          const response = await apiClient.getCurrentUser();
          if (response.data) {
            setUser(response.data);
          } else {
            // Token is invalid, clear it
            await signOut();
          }
        } catch (error) {
          console.error('Error checking auth status:', error);
          await signOut();
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, []);

  const signIn = async (username: string, password: string): Promise<{ error?: string }> => {
    try {
      const response = await apiClient.login(username, password);
      
      if (response.error) {
        return { error: response.error };
      }

      // Get user info after successful login
      const userResponse = await apiClient.getCurrentUser();
      if (userResponse.data) {
        setUser(userResponse.data);
      }

      return {};
    } catch (error) {
      console.error('Sign in error:', error);
      return { error: 'An unexpected error occurred' };
    }
  };

  const signUp = async (userData: {
    username: string;
    email: string;
    password: string;
    confirm_password: string;
  }): Promise<{ error?: string }> => {
    try {
      const response = await apiClient.signup(userData);
      
      if (response.error) {
        return { error: response.error };
      }

      // After successful signup, automatically sign in
      const loginResponse = await signIn(userData.username, userData.password);
      return loginResponse;
    } catch (error) {
      console.error('Sign up error:', error);
      return { error: 'An unexpected error occurred during signup' };
    }
  };

  const signOut = async () => {
    try {
      await apiClient.logout();
    } catch (error) {
      console.error('Error signing out:', error);
    } finally {
      setUser(null);
    }
  };

  const value = {
    user,
    loading,
    signIn,
    signUp,
    signOut,
    isAuthenticated: !!user,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};