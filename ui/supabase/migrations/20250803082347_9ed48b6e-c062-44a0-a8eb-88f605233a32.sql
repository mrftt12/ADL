-- Create enum types for various statuses and configurations
CREATE TYPE public.project_status AS ENUM ('draft', 'running', 'completed', 'failed', 'paused');
CREATE TYPE public.experiment_status AS ENUM ('queued', 'running', 'completed', 'failed', 'paused');
CREATE TYPE public.model_type AS ENUM ('classification', 'regression', 'object_detection', 'nlp', 'time_series');
CREATE TYPE public.optimizer_type AS ENUM ('adam', 'sgd', 'rmsprop', 'adamw');

-- Create profiles table for user information
CREATE TABLE public.profiles (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create projects table
CREATE TABLE public.projects (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  status public.project_status NOT NULL DEFAULT 'draft',
  model_type public.model_type NOT NULL,
  dataset_name TEXT NOT NULL,
  dataset_size INTEGER,
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  best_accuracy DECIMAL(5,4),
  runtime_minutes INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create experiments table for individual AutoML runs
CREATE TABLE public.experiments (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status public.experiment_status NOT NULL DEFAULT 'queued',
  architecture_config JSONB,
  hyperparameters JSONB,
  metrics JSONB,
  progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
  accuracy DECIMAL(5,4),
  loss DECIMAL(10,6),
  runtime_seconds INTEGER DEFAULT 0,
  started_at TIMESTAMP WITH TIME ZONE,
  completed_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create datasets table
CREATE TABLE public.datasets (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT,
  file_path TEXT,
  file_size INTEGER,
  num_samples INTEGER,
  num_features INTEGER,
  target_column TEXT,
  data_types JSONB,
  preprocessing_config JSONB,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create architecture_search table for NAS results
CREATE TABLE public.architecture_search (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  experiment_id UUID NOT NULL REFERENCES public.experiments(id) ON DELETE CASCADE,
  architecture JSONB NOT NULL,
  performance_score DECIMAL(10,6),
  parameters_count INTEGER,
  flops BIGINT,
  search_iteration INTEGER,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Create hyperparameter_search table
CREATE TABLE public.hyperparameter_search (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  experiment_id UUID NOT NULL REFERENCES public.experiments(id) ON DELETE CASCADE,
  learning_rate DECIMAL(10,8),
  batch_size INTEGER,
  optimizer public.optimizer_type,
  epochs INTEGER,
  dropout_rate DECIMAL(3,2),
  weight_decay DECIMAL(10,8),
  performance_score DECIMAL(10,6),
  search_iteration INTEGER,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.architecture_search ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hyperparameter_search ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for profiles
CREATE POLICY "Users can view their own profile" 
ON public.profiles 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can update their own profile" 
ON public.profiles 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own profile" 
ON public.profiles 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- Create RLS policies for projects
CREATE POLICY "Users can view their own projects" 
ON public.projects 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own projects" 
ON public.projects 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own projects" 
ON public.projects 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects" 
ON public.projects 
FOR DELETE 
USING (auth.uid() = user_id);

-- Create RLS policies for experiments
CREATE POLICY "Users can view experiments for their projects" 
ON public.experiments 
FOR SELECT 
USING (EXISTS (
  SELECT 1 FROM public.projects 
  WHERE projects.id = experiments.project_id 
  AND projects.user_id = auth.uid()
));

CREATE POLICY "Users can create experiments for their projects" 
ON public.experiments 
FOR INSERT 
WITH CHECK (EXISTS (
  SELECT 1 FROM public.projects 
  WHERE projects.id = experiments.project_id 
  AND projects.user_id = auth.uid()
));

CREATE POLICY "Users can update experiments for their projects" 
ON public.experiments 
FOR UPDATE 
USING (EXISTS (
  SELECT 1 FROM public.projects 
  WHERE projects.id = experiments.project_id 
  AND projects.user_id = auth.uid()
));

CREATE POLICY "Users can delete experiments for their projects" 
ON public.experiments 
FOR DELETE 
USING (EXISTS (
  SELECT 1 FROM public.projects 
  WHERE projects.id = experiments.project_id 
  AND projects.user_id = auth.uid()
));

-- Create RLS policies for datasets
CREATE POLICY "Users can view their own datasets" 
ON public.datasets 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own datasets" 
ON public.datasets 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own datasets" 
ON public.datasets 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own datasets" 
ON public.datasets 
FOR DELETE 
USING (auth.uid() = user_id);

-- Create RLS policies for architecture_search
CREATE POLICY "Users can view architecture search for their experiments" 
ON public.architecture_search 
FOR SELECT 
USING (EXISTS (
  SELECT 1 FROM public.experiments e
  JOIN public.projects p ON e.project_id = p.id
  WHERE e.id = architecture_search.experiment_id 
  AND p.user_id = auth.uid()
));

CREATE POLICY "Users can create architecture search for their experiments" 
ON public.architecture_search 
FOR INSERT 
WITH CHECK (EXISTS (
  SELECT 1 FROM public.experiments e
  JOIN public.projects p ON e.project_id = p.id
  WHERE e.id = architecture_search.experiment_id 
  AND p.user_id = auth.uid()
));

-- Create RLS policies for hyperparameter_search
CREATE POLICY "Users can view hyperparameter search for their experiments" 
ON public.hyperparameter_search 
FOR SELECT 
USING (EXISTS (
  SELECT 1 FROM public.experiments e
  JOIN public.projects p ON e.project_id = p.id
  WHERE e.id = hyperparameter_search.experiment_id 
  AND p.user_id = auth.uid()
));

CREATE POLICY "Users can create hyperparameter search for their experiments" 
ON public.hyperparameter_search 
FOR INSERT 
WITH CHECK (EXISTS (
  SELECT 1 FROM public.experiments e
  JOIN public.projects p ON e.project_id = p.id
  WHERE e.id = hyperparameter_search.experiment_id 
  AND p.user_id = auth.uid()
));

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_projects_updated_at
  BEFORE UPDATE ON public.projects
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at
  BEFORE UPDATE ON public.experiments
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at
  BEFORE UPDATE ON public.datasets
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Create function to handle new user profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (user_id, display_name)
  VALUES (NEW.id, NEW.raw_user_meta_data ->> 'display_name');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger for automatic profile creation
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-- Create indexes for better performance
CREATE INDEX idx_projects_user_id ON public.projects(user_id);
CREATE INDEX idx_projects_status ON public.projects(status);
CREATE INDEX idx_experiments_project_id ON public.experiments(project_id);
CREATE INDEX idx_experiments_status ON public.experiments(status);
CREATE INDEX idx_datasets_user_id ON public.datasets(user_id);
CREATE INDEX idx_architecture_search_experiment_id ON public.architecture_search(experiment_id);
CREATE INDEX idx_hyperparameter_search_experiment_id ON public.hyperparameter_search(experiment_id);