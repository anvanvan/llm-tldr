// Fixture: export const with arrow function and function_expression forms
// Tests: _extract_ts_function + _get_variable_declarator_name

export const useEventFormSubmission = (form: HTMLFormElement, opts: SubmitOptions): void => {
    form.submit();
};

export const enumerateCachedWindowKeys = function (): string[] {
    return Object.keys(window);
};
