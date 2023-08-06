htmx.on("htmx:afterRequest", function(evt) {
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

});


document.addEventListener('DOMContentLoaded', function() {
  // initialize all modals
  const allModalElems = document.querySelectorAll('.modal')
  const allModals = M.Modal.init(allModalElems, {});
});

