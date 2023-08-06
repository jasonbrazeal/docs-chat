htmx.on("htmx:afterRequest", function(evt) {
    if (evt.detail.elt.id === "api-key-submit") {
      const apiKeyForm = document.getElementById('api-key-form');
      apiKeyForm.reset();
    }
    console.log(evt.detail.elt);
    console.log("The element that dispatched the request: ", evt.detail.elt);
    console.log("The XMLHttpRequest: ", evt.detail.xhr);
    console.log("The target of the request: ", evt.detail.target);
    console.log("The configuration of the AJAX request: ", evt.detail.requestConfig);
    console.log("The event that triggered the request: ", evt.detail.requestConfig.triggeringEvent);
    console.log("True if the response has a 20x status code or is marked detail.isError = false in the htmx:beforeSwap event, else false: ", evt.detail.successful);
    console.log("true if the response does not have a 20x status code or is marked detail.isError = true in the htmx:beforeSwap event, else false: ", evt.detail.failed);
});

htmx.on("htmx:onLoadError", function(evt) {
    // htmx.find("#error-div").innerHTML = "A network error occured...";
    console.log(evt)
    console.log("ERROR!")
  }
)

htmx.on("htmx:beforeRequest", function(evt) {

  // settings
  const apiKeyInput = document.getElementById('api-key-input');
  if (apiKeyInput) {
    if (!apiKeyInput.value) {
      evt.preventDefault()
    }
  }

});


document.addEventListener('DOMContentLoaded', function() {

  // settings
  const apiKeyModalElem = document.getElementById('api-key-modal');
  const apiKeyModal = M.Modal.init(apiKeyModalElem, {
    onOpenEnd: () => {
      const apiKeyInput = document.getElementById('api-key-input');
      apiKeyInput.focus();
    },
    onCloseEnd: () => {
      const apiKeyForm = document.getElementById('api-key-form');
      apiKeyForm.reset();
    }
  });

  // settings
  const apiKeyForm = document.getElementById('api-key-form');
  if (apiKeyForm) {
    apiKeyForm.addEventListener('keypress', (e) => {
      if (e.code === 'Enter') {
        e.preventDefault();
        const apiKeySubmit = document.getElementById('api-key-submit');
        apiKeySubmit.click();
      }
    });
  }



});
