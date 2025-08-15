-- Create a special policy to allow the experiment runner function to access experiments
-- This allows updates when the function is running in service context
CREATE POLICY "Allow service role to manage experiments" 
ON public.experiments 
FOR ALL 
TO service_role
USING (true)
WITH CHECK (true);

-- Also allow authenticated users to read all experiments for the runner context
-- We'll use a more permissive policy temporarily for the demo
CREATE POLICY "Allow experiment processing" 
ON public.experiments 
FOR ALL 
TO authenticated, anon
USING (status = 'queued' OR status = 'running')
WITH CHECK (status = 'queued' OR status = 'running');