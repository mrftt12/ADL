-- Fix the search path security issue for the function
CREATE OR REPLACE FUNCTION public.process_queued_experiments()
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    experiment_record RECORD;
BEGIN
    -- Get the oldest queued experiment
    SELECT * INTO experiment_record 
    FROM public.experiments 
    WHERE status = 'queued' 
    ORDER BY created_at ASC 
    LIMIT 1;
    
    -- If there's a queued experiment, start it
    IF experiment_record.id IS NOT NULL THEN
        UPDATE public.experiments 
        SET 
            status = 'running',
            started_at = now(),
            updated_at = now()
        WHERE id = experiment_record.id;
    END IF;
END;
$$;