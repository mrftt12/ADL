export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instanciate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "12.2.3 (519615d)"
  }
  public: {
    Tables: {
      architecture_search: {
        Row: {
          architecture: Json
          created_at: string
          experiment_id: string
          flops: number | null
          id: string
          parameters_count: number | null
          performance_score: number | null
          search_iteration: number | null
        }
        Insert: {
          architecture: Json
          created_at?: string
          experiment_id: string
          flops?: number | null
          id?: string
          parameters_count?: number | null
          performance_score?: number | null
          search_iteration?: number | null
        }
        Update: {
          architecture?: Json
          created_at?: string
          experiment_id?: string
          flops?: number | null
          id?: string
          parameters_count?: number | null
          performance_score?: number | null
          search_iteration?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "architecture_search_experiment_id_fkey"
            columns: ["experiment_id"]
            isOneToOne: false
            referencedRelation: "experiments"
            referencedColumns: ["id"]
          },
        ]
      }
      datasets: {
        Row: {
          created_at: string
          data_types: Json | null
          description: string | null
          file_path: string | null
          file_size: number | null
          id: string
          name: string
          num_features: number | null
          num_samples: number | null
          preprocessing_config: Json | null
          target_column: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          data_types?: Json | null
          description?: string | null
          file_path?: string | null
          file_size?: number | null
          id?: string
          name: string
          num_features?: number | null
          num_samples?: number | null
          preprocessing_config?: Json | null
          target_column?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          data_types?: Json | null
          description?: string | null
          file_path?: string | null
          file_size?: number | null
          id?: string
          name?: string
          num_features?: number | null
          num_samples?: number | null
          preprocessing_config?: Json | null
          target_column?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      experiments: {
        Row: {
          accuracy: number | null
          architecture_config: Json | null
          completed_at: string | null
          created_at: string
          hyperparameters: Json | null
          id: string
          loss: number | null
          metrics: Json | null
          name: string
          progress: number | null
          project_id: string
          runtime_seconds: number | null
          started_at: string | null
          status: Database["public"]["Enums"]["experiment_status"]
          updated_at: string
        }
        Insert: {
          accuracy?: number | null
          architecture_config?: Json | null
          completed_at?: string | null
          created_at?: string
          hyperparameters?: Json | null
          id?: string
          loss?: number | null
          metrics?: Json | null
          name: string
          progress?: number | null
          project_id: string
          runtime_seconds?: number | null
          started_at?: string | null
          status?: Database["public"]["Enums"]["experiment_status"]
          updated_at?: string
        }
        Update: {
          accuracy?: number | null
          architecture_config?: Json | null
          completed_at?: string | null
          created_at?: string
          hyperparameters?: Json | null
          id?: string
          loss?: number | null
          metrics?: Json | null
          name?: string
          progress?: number | null
          project_id?: string
          runtime_seconds?: number | null
          started_at?: string | null
          status?: Database["public"]["Enums"]["experiment_status"]
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "experiments_project_id_fkey"
            columns: ["project_id"]
            isOneToOne: false
            referencedRelation: "projects"
            referencedColumns: ["id"]
          },
        ]
      }
      hyperparameter_search: {
        Row: {
          batch_size: number | null
          created_at: string
          dropout_rate: number | null
          epochs: number | null
          experiment_id: string
          id: string
          learning_rate: number | null
          optimizer: Database["public"]["Enums"]["optimizer_type"] | null
          performance_score: number | null
          search_iteration: number | null
          weight_decay: number | null
        }
        Insert: {
          batch_size?: number | null
          created_at?: string
          dropout_rate?: number | null
          epochs?: number | null
          experiment_id: string
          id?: string
          learning_rate?: number | null
          optimizer?: Database["public"]["Enums"]["optimizer_type"] | null
          performance_score?: number | null
          search_iteration?: number | null
          weight_decay?: number | null
        }
        Update: {
          batch_size?: number | null
          created_at?: string
          dropout_rate?: number | null
          epochs?: number | null
          experiment_id?: string
          id?: string
          learning_rate?: number | null
          optimizer?: Database["public"]["Enums"]["optimizer_type"] | null
          performance_score?: number | null
          search_iteration?: number | null
          weight_decay?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "hyperparameter_search_experiment_id_fkey"
            columns: ["experiment_id"]
            isOneToOne: false
            referencedRelation: "experiments"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string
          display_name: string | null
          id: string
          updated_at: string
          user_id: string
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          id?: string
          updated_at?: string
          user_id: string
        }
        Update: {
          avatar_url?: string | null
          created_at?: string
          display_name?: string | null
          id?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      projects: {
        Row: {
          best_accuracy: number | null
          created_at: string
          dataset_name: string
          dataset_size: number | null
          description: string | null
          id: string
          model_type: Database["public"]["Enums"]["model_type"]
          name: string
          progress: number | null
          runtime_minutes: number | null
          status: Database["public"]["Enums"]["project_status"]
          updated_at: string
          user_id: string
        }
        Insert: {
          best_accuracy?: number | null
          created_at?: string
          dataset_name: string
          dataset_size?: number | null
          description?: string | null
          id?: string
          model_type: Database["public"]["Enums"]["model_type"]
          name: string
          progress?: number | null
          runtime_minutes?: number | null
          status?: Database["public"]["Enums"]["project_status"]
          updated_at?: string
          user_id: string
        }
        Update: {
          best_accuracy?: number | null
          created_at?: string
          dataset_name?: string
          dataset_size?: number | null
          description?: string | null
          id?: string
          model_type?: Database["public"]["Enums"]["model_type"]
          name?: string
          progress?: number | null
          runtime_minutes?: number | null
          status?: Database["public"]["Enums"]["project_status"]
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      process_queued_experiments: {
        Args: Record<PropertyKey, never>
        Returns: undefined
      }
    }
    Enums: {
      experiment_status:
        | "queued"
        | "running"
        | "completed"
        | "failed"
        | "paused"
      model_type:
        | "classification"
        | "regression"
        | "object_detection"
        | "nlp"
        | "time_series"
      optimizer_type: "adam" | "sgd" | "rmsprop" | "adamw"
      project_status: "draft" | "running" | "completed" | "failed" | "paused"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      experiment_status: ["queued", "running", "completed", "failed", "paused"],
      model_type: [
        "classification",
        "regression",
        "object_detection",
        "nlp",
        "time_series",
      ],
      optimizer_type: ["adam", "sgd", "rmsprop", "adamw"],
      project_status: ["draft", "running", "completed", "failed", "paused"],
    },
  },
} as const
