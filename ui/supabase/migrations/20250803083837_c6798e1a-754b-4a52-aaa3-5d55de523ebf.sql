-- Create storage bucket for datasets
INSERT INTO storage.buckets (id, name, public) VALUES ('datasets', 'datasets', false);

-- Create storage policies for datasets
CREATE POLICY "Users can view their own dataset files" 
ON storage.objects 
FOR SELECT 
USING (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can upload their own dataset files" 
ON storage.objects 
FOR INSERT 
WITH CHECK (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can update their own dataset files" 
ON storage.objects 
FOR UPDATE 
USING (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete their own dataset files" 
ON storage.objects 
FOR DELETE 
USING (bucket_id = 'datasets' AND auth.uid()::text = (storage.foldername(name))[1]);